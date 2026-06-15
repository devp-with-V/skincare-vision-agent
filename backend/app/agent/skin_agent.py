import os
import json
import logging
from typing import Dict, Any
import anthropic

from app.agent.prompts import SYSTEM_PROMPT, compile_user_prompt

logger = logging.getLogger("skincare-vision-backend")

class SkinAgent:
    def __init__(self):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.api_key:
            self.client = anthropic.AsyncAnthropic(api_key=self.api_key)
            logger.info("Anthropic Claude API client initialized successfully.")
        else:
            logger.warning(
                "ANTHROPIC_API_KEY environment variable is missing. "
                "The Skin Care Advisor LLM agent will run in Mock Fallback mode."
            )

    async def generate_analysis(self, overall_severity: float, regions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates skin analysis and recommendations.
        Calls Claude API if ANTHROPIC_API_KEY is present, otherwise falls back to a highly realistic mock routine generator.
        """
        # Compile prompt
        user_prompt = compile_user_prompt(overall_severity, regions_data)

        if self.client is not None:
            return await self._call_claude_api(user_prompt)
        else:
            return self._generate_mock_agent_response(overall_severity, regions_data)

    async def _call_claude_api(self, user_prompt: str) -> Dict[str, Any]:
        """Call Claude 3.5 Sonnet to perform structured analysis."""
        try:
            # We use Claude 3.5 Sonnet for the highest reasoning quality
            message = await self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1500,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse text response
            raw_text = message.content[0].text.strip()
            
            # Clean up markdown if Claude wraps it in ```json ... ```
            if raw_text.startswith("```"):
                lines = raw_text.split("\n")
                if lines[0].startswith("```json"):
                    raw_text = "\n".join(lines[1:-1]).strip()
                elif lines[0].startswith("```"):
                    raw_text = "\n".join(lines[1:-1]).strip()

            data = json.loads(raw_text)
            return data
            
        except Exception as e:
            logger.error(f"Error calling Claude API: {str(e)}. Falling back to mock generator.")
            return self._generate_mock_agent_response(0.5, {}) # Safe fallback

    def _generate_mock_agent_response(self, overall_severity: float, regions_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates realistic, tailored skin care routines based on scanning inputs.
        Used as a high-quality fallback when no Claude API key is supplied.
        """
        # 1. Determine the primary concern
        concerns = [data.get("dominant_concern") for data in regions_data.values() if data.get("dominant_concern")]
        primary_concern = "general"
        if concerns:
            counts = {c: concerns.count(c) for c in set(concerns)}
            primary_concern = max(counts, key=counts.get)
            
        # Determine dermatologist referral
        needs_derm = overall_severity > 0.5
        derm_reason = (
            f"We detected moderate-to-severe concern clusters ({int(overall_severity * 100)}% overall rating) "
            "which are best evaluated by a certified dermatologist for prescription treatments."
            if needs_derm else None
        )

        # 2. Build tailored routines
        if primary_concern == "acne":
            condition_name = "Mild-to-Moderate Acne vulgaris"
            condition_desc = (
                "Your scan shows localized inflammatory lesions (acne) primarily concentrated on the cheeks. "
                "Acne is caused by blocked pores, sebum overproduction, and acne bacteria. "
                "The suggested routine focuses on salicylic acid to clear pore debris, and niacinamide to calm inflammation."
            )
            routine = [
                {
                    "step_number": 1,
                    "time_of_day": "AM",
                    "step_name": "Cleanser",
                    "ingredients": ["Salicylic Acid 2%", "Centella Asiatica"],
                    "instructions": "Wash gently with lukewarm water. Do not scrub, as it will worsen inflammation."
                },
                {
                    "step_number": 2,
                    "time_of_day": "AM",
                    "step_name": "Hydrate & Treat",
                    "ingredients": ["Niacinamide 5%", "Hyaluronic Acid"],
                    "instructions": "Apply 3-4 drops to calm redness and rebuild the skin barrier."
                },
                {
                    "step_number": 3,
                    "time_of_day": "AM",
                    "step_name": "Protect",
                    "ingredients": ["Broad-Spectrum SPF 50+"],
                    "instructions": "Apply generously. UV rays can darken post-acne spots (hyperpigmentation)."
                },
                {
                    "step_number": 1,
                    "time_of_day": "PM",
                    "step_name": "Cleanse",
                    "ingredients": ["Gentle Foaming Cleanser"],
                    "instructions": "Double cleanse if you wore heavy SPF or were outdoors in dirt/pollution."
                },
                {
                    "step_number": 2,
                    "time_of_day": "PM",
                    "step_name": "Treat",
                    "ingredients": ["Salicylic Acid 2% Serum"],
                    "instructions": "Apply to acne-prone zones to unclog pores overnight. Use 3 times a week."
                },
                {
                    "step_number": 3,
                    "time_of_day": "PM",
                    "step_name": "Moisturize",
                    "ingredients": ["Ceramides", "Squalane"],
                    "instructions": "Apply a non-comedogenic barrier support cream to lock in moisture."
                }
            ]
            lifestyle_tips = [
                "Avoid squeezing or popping blemishes to prevent permanent scarring and bacterial spread.",
                "Change your pillowcases every 2-3 days to reduce oil build-up.",
                "Incorporate a light, water-based gel moisturizer instead of heavy oils."
            ]
            
        elif primary_concern == "dryness_patch":
            condition_name = "Dehydrated & Dry Skin Barrier"
            condition_desc = (
                "Dryness and flaky patches were detected, notably around the forehead or chin. "
                "This indicates a depletion of the skin's lipid barrier. "
                "The routine focuses on ceramides and humectants like hyaluronic acid to bind moisture."
            )
            routine = [
                {
                    "step_number": 1,
                    "time_of_day": "AM",
                    "step_name": "Cleanse",
                    "ingredients": ["Hydrating Cream Cleanser"],
                    "instructions": "Use a non-foaming hydrating wash, or simply rinse with cold water."
                },
                {
                    "step_number": 2,
                    "time_of_day": "AM",
                    "step_name": "Hydrate",
                    "ingredients": ["Hyaluronic Acid 2%", "Glycerin"],
                    "instructions": "Apply onto damp skin to lock in maximum water hydration."
                },
                {
                    "step_number": 3,
                    "time_of_day": "AM",
                    "step_name": "Moisturize & Protect",
                    "ingredients": ["Ceramides Cream", "Zinc Oxide SPF 30+"],
                    "instructions": "Use a rich cream to seal the barrier, followed by mineral SPF."
                },
                {
                    "step_number": 1,
                    "time_of_day": "PM",
                    "step_name": "Cleanse",
                    "ingredients": ["Cleansing Balm / Cream Wash"],
                    "instructions": "Gently dissolve sunscreen and debris without stripping the lipid barrier."
                },
                {
                    "step_number": 2,
                    "time_of_day": "PM",
                    "step_name": "Repair & Seal",
                    "ingredients": ["Ceramides", "Panthenol (Vitamin B5)"],
                    "instructions": "Apply a thick, nourishing barrier repair cream."
                }
            ]
            lifestyle_tips = [
                "Drink at least 2.5 liters of water daily to maintain cellular hydration.",
                "Avoid hot showers, as hot water strips natural oils from the skin face barrier.",
                "Use a humidifier in dry indoor environments or air-conditioned rooms."
            ]
            
        elif primary_concern == "redness":
            condition_name = "Skin Erythema & Vascular Redness"
            condition_desc = (
                "Localized redness or flushing was detected, suggesting high sensitivity or mild rosacea. "
                "The focus is on calming ingredients that reduce blood vessel dilation and skin sensitivity."
            )
            routine = [
                {
                    "step_number": 1,
                    "time_of_day": "AM",
                    "time_of_day": "AM",
                    "step_name": "Cleanse",
                    "ingredients": ["Centella Asiatica (Cica) Gel Cleanser"],
                    "instructions": "Use an ultra-gentle, pH-balanced soothing gel wash."
                },
                {
                    "step_number": 2,
                    "time_of_day": "AM",
                    "step_name": "Calm",
                    "ingredients": ["Centella Asiatica Extract", "Panthenol"],
                    "instructions": "Apply a soothing cica serum to reduce thermal heat and irritation."
                },
                {
                    "step_number": 3,
                    "time_of_day": "AM",
                    "step_name": "Protect",
                    "ingredients": ["Mineral SPF 50+ (Zinc Oxide)"],
                    "instructions": "Zinc oxide is naturally anti-inflammatory and calms flushed skin."
                },
                {
                    "step_number": 1,
                    "time_of_day": "PM",
                    "step_name": "Cleanse",
                    "ingredients": ["Soothing Milk Cleanser"],
                    "instructions": "Wipe away build-up without friction. Avoid using face scrubs or washcloths."
                },
                {
                    "step_number": 2,
                    "time_of_day": "PM",
                    "step_name": "Treat & Repair",
                    "ingredients": ["Niacinamide 2%", "Allantoin"],
                    "instructions": "Apply a light skin-barrier repair serum."
                }
            ]
            lifestyle_tips = [
                "Avoid spicy foods, caffeine, and alcohol, as they are known triggers for skin flushing.",
                "Keep skincare products in a beauty fridge; cold application constricts capillaries.",
                "Wear broad-brimmed hats outdoors to shelter sensitive skin from UV-induced heat."
            ]
            
        else: # general or dark_spots
            condition_name = "Mild Hyperpigmentation / Tone Unevenness"
            condition_desc = (
                "Your scan shows minor tone unevenness or hyperpigmentation blemishes. "
                "Hyperpigmentation is caused by localized melanin overproduction. "
                "Vitamin C and sunscreen prevent UV activation, while niacinamide blocks pigment transfer."
            )
            routine = [
                {
                    "step_number": 1,
                    "time_of_day": "AM",
                    "step_name": "Cleanse",
                    "ingredients": ["Gentle Cleansing Gel"],
                    "instructions": "Wash to prep skin for antioxidant absorption."
                },
                {
                    "step_number": 2,
                    "time_of_day": "AM",
                    "step_name": "Brighten",
                    "ingredients": ["Vitamin C (L-Ascorbic Acid)", "Ferulic Acid"],
                    "instructions": "Apply 3 drops. Vitamin C is a potent antioxidant that fades dark spots."
                },
                {
                    "step_number": 3,
                    "time_of_day": "AM",
                    "step_name": "Protect",
                    "ingredients": ["Broad-Spectrum SPF 50+"],
                    "instructions": "Essential step. UV rays will immediately re-darken fading spots."
                },
                {
                    "step_number": 1,
                    "time_of_day": "PM",
                    "step_name": "Cleanse",
                    "ingredients": ["Foaming Cleanser"],
                    "instructions": "Cleanse thoroughly to remove daily sunscreen and pollution."
                },
                {
                    "step_number": 2,
                    "time_of_day": "PM",
                    "step_name": "Treat",
                    "ingredients": ["Retinol 0.2%", "Niacinamide"],
                    "instructions": "Apply retinol to boost skin cell turnover and fade pigment. Use 3 nights a week."
                }
            ]
            lifestyle_tips = [
                "Incorporate a physical sunscreen (zinc oxide/titanium dioxide) to protect skin pigment from UV.",
                "Ensure you apply SPF even on cloudy days, as UVA rays pass through clouds.",
                "Incorporate foods rich in antioxidants, like berries and green tea, into your diet."
            ]

        return {
            "condition_name": condition_name,
            "condition_desc": condition_desc,
            "overall_summary": f"Based on an overall severity score of {overall_severity * 100:.0f}%, the scan identified '{primary_concern}' as the dominant concern.",
            "dermatologist_flag": needs_derm,
            "dermatologist_reason": derm_reason,
            "disclaimer": "This analysis is generated by an AI assistant for educational purposes only. It is not medical advice. Consult a dermatologist for any persistent or severe conditions.",
            "routine": routine,
            "lifestyle_tips": lifestyle_tips
        }
