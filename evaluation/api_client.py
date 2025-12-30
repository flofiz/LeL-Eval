"""
Client API asynchrone pour VLLM.

Ce module fournit les fonctions pour communiquer avec l'API VLLM
de manière asynchrone, permettant le traitement parallèle des échantillons.
"""

import os
import json
import asyncio
import aiohttp
from typing import Dict

from .tsv_parser import extract_tsv_from_response
from .labelme_export import tsv_to_labelme


def image_to_url(base64_image: str) -> str:
    """
    Convertit une image base64 en URL data.
    
    Args:
        base64_image: Image encodée en base64
        
    Returns:
        URL data:image/jpeg;base64,...
    """
    return f"data:image/jpeg;base64,{base64_image}"


# Regex pour limiter à Latin-1 (évite les caractères chinois)
LATIN1_REGEX = r"[\x09\x0A\x0D\x20-\x7E\xA0-\xFF]*"


async def call_vllm_api(session: aiohttp.ClientSession,
                       api_url: str,
                       system_prompt: str,
                       user_prompt: str,
                       image_base64: str,
                       temperature: float = 0.0,
                       max_tokens: int = 2048,
                       use_guided_regex: bool = True) -> Dict:
    """
    Envoie une requête asynchrone à l'API VLLM (format OpenAI).
    
    Args:
        session: Session aiohttp
        api_url: URL de base de l'API VLLM
        system_prompt: Prompt système
        user_prompt: Prompt utilisateur
        image_base64: Image encodée en base64
        temperature: Température de génération (défaut: 0.0 pour déterministe)
        max_tokens: Nombre maximum de tokens à générer
        use_guided_regex: Si True, utilise le regex Latin-1 pour éviter les caractères non-latins
        
    Returns:
        Dict avec 'content' (texte), 'logprobs' (liste), et 'perplexity' (float ou None)
    """
    import math
    
    # Construire le message avec l'image
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_to_url(image_base64)
                    }
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]
    
    payload = {
        "model": "Qwen3-VL-4B",  # Nom du modèle servi
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 1.0,
        "logprobs": True,  # Activer les logprobs pour calculer la perplexité
        "top_logprobs": 1
    }
    
    # Ajouter le guided_regex si activé (extra_body pour compatibilité OpenAI)
    if use_guided_regex:
        payload["guided_regex"] = LATIN1_REGEX
    
    try:
        async with session.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message']['content']
                
                # Extraire les logprobs si disponibles
                logprobs_data = None
                perplexity = None
                
                choice = result['choices'][0]
                if 'logprobs' in choice and choice['logprobs'] is not None:
                    logprobs_content = choice['logprobs'].get('content', [])
                    if logprobs_content:
                        # Calculer la perplexité: exp(-moyenne des log probs)
                        log_probs = []
                        for token_info in logprobs_content:
                            if 'logprob' in token_info:
                                log_probs.append(token_info['logprob'])
                        
                        if log_probs:
                            avg_log_prob = sum(log_probs) / len(log_probs)
                            perplexity = math.exp(-avg_log_prob)
                            logprobs_data = log_probs
                
                return {
                    'content': content,
                    'logprobs': logprobs_data,
                    'perplexity': perplexity
                }
            else:
                error_text = await response.text()
                print(f"Erreur API (status {response.status}): {error_text}")
                return {'content': '', 'logprobs': None, 'perplexity': None}
    except asyncio.TimeoutError:
        print("Timeout lors de l'appel API")
        return {'content': '', 'logprobs': None, 'perplexity': None}
    except Exception as e:
        print(f"Erreur lors de l'appel API: {e}")
        return {'content': '', 'logprobs': None, 'perplexity': None}


async def process_sample(session: aiohttp.ClientSession,
                        sample: Dict,
                        api_url: str,
                        system_prompt: str,
                        user_prompt: str,
                        semaphore: asyncio.Semaphore,
                        output_dir: str = None) -> Dict:
    """
    Traite un échantillon de manière asynchrone et sauvegarde au format LabelMe.
    
    Args:
        session: Session aiohttp
        sample: Échantillon à traiter (dict avec 'name', 'image', 'tsv')
        api_url: URL de l'API VLLM
        system_prompt: Prompt système
        user_prompt: Prompt utilisateur
        semaphore: Sémaphore pour limiter la concurrence
        output_dir: Dossier de sortie pour les fichiers LabelMe (optionnel)
        
    Returns:
        Dictionnaire avec les résultats (name, predicted, ground_truth, full_response, perplexity)
    """
    async with semaphore:
        # Appeler l'API
        api_response = await call_vllm_api(
            session,
            api_url,
            system_prompt,
            user_prompt,
            sample['image']
        )
        
        response_text = api_response['content']
        perplexity = api_response['perplexity']
        
        # Extraire le TSV de la réponse
        predicted_tsv = extract_tsv_from_response(response_text)
        
        # Créer les fichiers LabelMe si output_dir est spécifié
        if output_dir:
            # Créer le sous-dossier labelme
            labelme_dir = os.path.join(output_dir, "labelme")
            os.makedirs(labelme_dir, exist_ok=True)
            
            # Prédiction
            pred_labelme = tsv_to_labelme(
                predicted_tsv,
                sample['image'],
                sample['name']
            )
            pred_filename = os.path.join(labelme_dir, f"{sample['name']}_pred.json")
            with open(pred_filename, 'w', encoding='utf-8') as f:
                json.dump(pred_labelme, f, indent=2, ensure_ascii=False)
            
            # Ground truth
            gt_labelme = tsv_to_labelme(
                sample['tsv'],
                sample['image'],
                sample['name']
            )
            gt_filename = os.path.join(labelme_dir, f"{sample['name']}_gt.json")
            with open(gt_filename, 'w', encoding='utf-8') as f:
                json.dump(gt_labelme, f, indent=2, ensure_ascii=False)
        
        return {
            'name': sample['name'],
            'predicted': predicted_tsv,
            'ground_truth': sample['tsv'],
            'full_response': response_text,
            'perplexity': perplexity
        }

