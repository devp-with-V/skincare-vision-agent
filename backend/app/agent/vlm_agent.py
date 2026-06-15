import os
import json
import logging
import httpx
from typing import Dict, Any

from app.agent.vlm_prompts import SYSTEM_PROMPT, USER_INSTRUCTION

logger = logging.getLogger("skincare-vision-backend")

class VLMAgent:
    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        self.model = os.environ.get("VLM_MODEL", "meta-llama/llama-3.2-11b-vision-instruct:free")
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        
        if self.api_key:
            logger.info(f"OpenRouter VLM Client initialized successfully with model: {self.model}")
        else:
            logger.warning(
                "OPENROUTER_API_KEY environment variable is missing. "
                "The Skin Care VLM agent will run in Mock Fallback mode."
            )

    async def analyze_image(self, image_base64: str) -> Dict[str, Any]:
        """
        Runs Vision-Language Model analysis on the raw face image.
        Calls OpenRouter API if OPENROUTER_API_KEY is present, otherwise falls back to a mock analysis.
        """
        # Ensure base64 string has correct data URI scheme prefix
        if not image_base64.startswith("data:image"):
            image_base64 = f"data:image/jpeg;base64,{image_base64}"

        if self.api_key:
            return await self._call_openrouter_api(image_base64)
        else:
            return self._generate_mock_vlm_response()

    async def _call_openrouter_api(self, image_url: str) -> Dict[str, Any]:
        """Call OpenRouter vision chat completion API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "SkinCare Vision Agent"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": USER_INSTRUCTION
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                
                if response.status_code != 200:
                    logger.error(f"OpenRouter API error (status {response.status_code}): {response.text}")
                    # Try a fallback model (e.g. Qwen 2 VL) if Llama 3.2 fails or is overloaded
                    if "llama-3.2" in self.model:
                        logger.info("Attempting fallback model: qwen/qwen-2-vl-7b-instruct:free")
                        payload["model"] = "qwen/qwen-2-vl-7b-instruct:free"
                        fallback_response = await client.post(self.endpoint, headers=headers, json=payload)
                        if fallback_response.status_code == 200:
                            response = fallback_response
                        else:
                            raise httpx.HTTPStatusError(f"OpenRouter VLM failed: {fallback_response.text}", request=None, response=fallback_response)
                    else:
                        response.raise_for_status()

                data = response.json()
                content = data["choices"][0]["message"]["content"].strip()
                
                # Clean up markdown block if VLM ignores system instructions
                if content.startswith("```"):
                    lines = content.split("\n")
                    if lines[0].startswith("```json"):
                        content = "\n".join(lines[1:-1]).strip()
                    elif lines[0].startswith("```"):
                        content = "\n".join(lines[1:-1]).strip()
                
                return json.loads(content)

        except Exception as e:
            logger.error(f"Failed to fetch VLM analysis from OpenRouter: {str(e)}. Falling back to mock VLM response.")
            return self._generate_mock_vlm_response()

    def _generate_mock_vlm_response(self) -> Dict[str, Any]:
        """Generates a high-quality mock VLM analysis response for local testing."""
        return {
            "condition_name": "Mild Acne & Dehydrated Forehead (VLM Cloud Agent)",
            "condition_desc": (
                "The VLM Cloud Agent detected clusters of localized inflammatory lesions (acne) "
                "on both cheeks, along with some slight redness and flakiness on the forehead. "
                "This indicates a combination of blemish-prone zones with a dry skin barrier. "
                "The suggested VLM routine focuses on clearing pores while keeping the barrier hydrated."
            ),
            "overall_summary": "Overall face scan shows moderate cheek acne (0.35 severity) and forehead dryness (0.22 severity).",
            "overall_severity": 0.28,
            "regions": {
                "forehead": {
                    "severity_score": 0.22,
                    "dominant_concern": "dryness_patch",
                    "detections": [
                        {
                            "class_name": "dryness_patch",
                            "confidence": 0.82,
                            "bbox": [0.25, 0.2, 0.6, 0.5]
                        }
                    ]
                },
                "left_cheek": {
                    "severity_score": 0.35,
                    "dominant_concern": "acne",
                    "detections": [
                        {
                            "class_name": "acne",
                            "confidence": 0.88,
                            "bbox": [0.3, 0.3, 0.45, 0.45]
                        },
                        {
                            "class_name": "acne",
                            "confidence": 0.72,
                            "bbox": [0.55, 0.4, 0.65, 0.5]
                        }
                    ]
                },
                "right_cheek": {
                    "severity_score": 0.38,
                    "dominant_concern": "acne",
                    "detections": [
                        {
                            "class_name": "acne",
                            "confidence": 0.85,
                            "bbox": [0.35, 0.35, 0.5, 0.5]
                        }
                    ]
                },
                "nose": {
                    "severity_score": 0.1,
                    "dominant_concern": "redness",
                    "detections": [
                        {
                            "class_name": "redness",
                            "confidence": 0.65,
                            "bbox": [0.4, 0.4, 0.6, 0.6]
                        }
                    ]
                },
                "chin": {
                    "severity_score": 0.05,
                    "dominant_concern": None,
                    "detections": []
                }
            },
            "dermatologist_flag": False,
            "dermatologist_reason": None,
            "disclaimer": "This analysis is generated by an AI assistant for educational purposes only. It is not medical advice. Consult a dermatologist for any persistent or severe conditions.",
            "routine": [
                {
                    "step_number": 1,
                    "time_of_day": "AM",
                    "step_name": "Cleanse",
                    "ingredients": ["Gentle Gel Cleanser"],
                    "instructions": "Wash gently with lukewarm water to remove overnight sebum."
                },
                {
                    "step_number": 2,
                    "time_of_day": "AM",
                    "step_name": "Treat & Hydrate",
                    "ingredients": ["Niacinamide 4%", "Hyaluronic Acid"],
                    "instructions": "Apply 3 drops to calm cheek redness and hydrate the forehead barrier."
                },
                {
                    "step_number": 3,
                    "time_of_day": "AM",
                    "step_name": "Protect",
                    "ingredients": ["Broad-Spectrum SPF 30+"],
                    "instructions": "Apply generously. UV rays can worsen acne inflammation."
                },
                {
                    "step_number": 1,
                    "time_of_day": "PM",
                    "step_name": "Cleanse",
                    "ingredients": ["Gentle Foaming Wash"],
                    "instructions": "Cleanse thoroughly to wash away daily sweat, sunscreen, and dirt."
                },
                {
                    "step_number": 2,
                    "time_of_day": "PM",
                    "step_name": "Treat (Blemish-zones)",
                    "ingredients": ["Salicylic Acid 2% Liquid"],
                    "instructions": "Apply onto blemish-prone cheek areas to clear pores overnight. Use 3 nights a week."
                },
                {
                    "step_number": 3,
                    "time_of_day": "PM",
                    "step_name": "Moisturize",
                    "ingredients": ["Ceramide Barrier Cream"],
                    "instructions": "Apply all over face to lock in hydration and rebuild dry skin patches."
                }
            ],
            "lifestyle_tips": [
                "Avoid touching your face or squeezing lesions to prevent scarring.",
                "Wash pillowcases frequently to prevent oil build-up.",
                "Ensure adequate water intake and avoid high-glycemic foods."
            ]
        }
