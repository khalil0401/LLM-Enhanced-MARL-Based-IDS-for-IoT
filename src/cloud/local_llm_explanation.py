"""
Cloud Layer: Local LLM Explanation Generator using Phi-3-mini
Replaces OpenAI GPT-4 with Microsoft Phi-3-mini-4k-instruct for local inference

Advantages:
- No API costs (100% free)
- Works offline (no internet needed after model download)
- Fast inference on single GPU
- Privacy-preserving (data never leaves Kaggle)
- Small model size (3.8B params, ~7GB)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional, List
from dataclasses import dataclass
import time


@dataclass
class LocalLLMConfig:
    """Configuration for local LLM explanation generator"""
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_new_tokens: int = 300
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True


class MITREMapper:
    """
    Maps attack types to MITRE ATT&CK for IoT taxonomy
    (Same as in original explanation_generator.py)
    """
    
    def __init__(self):
        self.attack_mapping = {
            0: {'id': 'T0000', 'tactic': 'Null', 'technique': 'No Attack'},
            1: {'id': 'T1498', 'tactic': 'Impact', 'technique': 'Network Denial of Service'},
            2: {'id': 'T1486', 'tactic': 'Impact', 'technique': 'Data Encrypted for Impact'},
            3: {'id': 'T1595', 'tactic': 'Reconnaissance', 'technique': 'Active Scanning'},
            4: {'id': 'T1041', 'tactic': 'Exfiltration', 'technique': 'Exfiltration Over C2 Channel'},
            5: {'id': 'T1059', 'tactic': 'Execution', 'technique': 'Command and Scripting Interpreter'},
            6: {'id': 'T1542', 'tactic': 'Persistence', 'technique': 'Pre-OS Boot: UEFI/BIOS'},
        }
    
    def map_attack(self, attack_type: int) -> Dict[str, str]:
        return self.attack_mapping.get(attack_type, self.attack_mapping[0])


class LocalLLMExplanationGenerator:
    """
    Phi-3-mini based explanation generator for IDS alerts
    
    Replaces GPT-4 API with local Microsoft Phi-3-mini model
    
    Model: microsoft/Phi-3-mini-4k-instruct
    Size: 3.8B parameters (~7GB)
    Context: 4K tokens
    Speed: ~50-100 tokens/sec on GPU
    
    Safety constraints:
    - Explanations are DESCRIPTIVE only
    - NO prescriptive actions
    - All recommendations require human authorization
    """
    
    def __init__(self, config: LocalLLMConfig):
        self.config = config
        self.mitre_mapper = MITREMapper()
        
        print(f"Loading Phi-3-mini model...")
        print(f"  Model: {config.model_name}")
        print(f"  Device: {config.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=True
        )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
            device_map=config.device,
            trust_remote_code=True
        )
        
        self.model.eval()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e9:.1f}B")
        print(f"   Memory: {torch.cuda.memory_allocated(0) / 1e9:.1f} GB" if config.device == "cuda" else "")
    
    def generate_explanation(self, alert_data: Dict) -> Dict[str, str]:
        """
        Generate human-readable explanation for alert using Phi-3-mini
        
        Args:
            alert_data: Dictionary with:
                - device_id: Device identifier
                - device_type: Type of IoT device
                - alert_level: 0-4 severity
                - attack_type: 0-6 attack category
                - confidence: 0-1 confidence score
                - features: Feature dictionary
                - timestamp: Alert timestamp
                
        Returns:
            Dictionary with explanation components
        """
        start_time = time.time()
        
        # Map to MITRE ATT&CK
        mitre_info = self.mitre_mapper.map_attack(alert_data['attack_type'])
        
        # Construct prompt
        prompt = self._build_prompt(alert_data, mitre_info)
        
        # Create messages (Phi-3 uses chat format)
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": prompt}
        ]
        
        # Generate explanation
        try:
            explanation_text = self._generate_text(messages)
        except Exception as e:
            print(f"Error generating explanation: {e}")
            explanation_text = self._get_fallback_explanation(alert_data)
        
        # Parse structured output
        result = self._parse_explanation(explanation_text, mitre_info)
        
        # Add metadata
        result['generation_time_ms'] = (time.time() - start_time) * 1000
        result['model'] = self.config.model_name
        
        return result
    
    def _generate_text(self, messages: List[Dict]) -> str:
        """
        Generate text using Phi-3-mini
        
        Args:
            messages: Chat-format messages
            
        Returns:
            Generated text
        """
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=3072  # Leave room for generation
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode (remove input prompt)
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()
    
    def _get_system_prompt(self) -> str:
        """System prompt defining LLM role and constraints"""
        return """You are a cybersecurity analyst assistant for IoT security.
Explain intrusion detection alerts to human analysts.

CRITICAL RULES:
- DESCRIPTIVE: Explain what was detected and why
- FACTUAL: Based only on evidence provided
- NO PRESCRIPTION: Never suggest blocking, quarantine, or mitigation
- HUMAN AUTHORITY: All actions need human approval
- NO SPECULATION: If uncertain, say so

Format:
1. Behavior Summary (2-3 sentences)
2. MITRE ATT&CK Context
3. Key Anomalies
4. Confidence Justification
5. Analyst Actions (observational only)"""
    
    def _build_prompt(self, alert_data: Dict, mitre_info: Dict) -> str:
        """Build user prompt with alert details"""
        
        attack_names = ["None", "DDoS", "Malware", "Reconnaissance", 
                       "Exfiltration", "Injection", "Firmware"]
        severity_names = ["Info", "Low", "Medium", "High", "Critical"]
        
        prompt = f"""IoT Network Alert:

Device: {alert_data['device_id']} ({alert_data.get('device_type', 'Unknown')})
Time: {alert_data.get('timestamp', 'N/A')}

Alert:
- Severity: {severity_names[alert_data['alert_level']]} ({alert_data['alert_level']}/4)
- Type: {attack_names[alert_data['attack_type']]}
- Confidence: {alert_data['confidence']:.2f}

MITRE ATT&CK:
- {mitre_info['id']}: {mitre_info['tactic']} - {mitre_info['technique']}

Anomalies:
{self._format_features(alert_data.get('features', {}))}

Provide concise explanation. Remember: descriptive only, human must approve actions."""
        
        return prompt
    
    def _format_features(self, features: Dict) -> str:
        """Format feature anomalies"""
        if not features:
            return "- No specific feature data"
        
        lines = []
        for key, value in list(features.items())[:5]:
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def _parse_explanation(self, text: str, mitre_info: Dict) -> Dict[str, str]:
        """Parse LLM output into structured format"""
        return {
            'explanation': text,
            'mitre_tactic': mitre_info['tactic'],
            'mitre_technique': mitre_info['technique'],
            'mitre_id': mitre_info['id'],
            'analyst_actions': self._extract_actions(text),
            'confidence_justification': self._extract_justification(text)
        }
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract analyst actions from text"""
        lines = text.split('\n')
        actions = []
        in_actions = False
        
        for line in lines:
            if 'Analyst Actions' in line or 'Recommended' in line:
                in_actions = True
                continue
            
            if in_actions and line.strip().startswith(('1.', '2.', '3.', '-', '•')):
                actions.append(line.strip())
        
        return actions if actions else ["Review device logs and network traffic"]
    
    def _extract_justification(self, text: str) -> str:
        """Extract confidence justification"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'Confidence' in line and i+1 < len(lines):
                return lines[i+1].strip()
        return "Based on observed behavioral anomalies"
    
    def _get_fallback_explanation(self, alert_data: Dict) -> str:
        """Fallback if generation fails"""
        return f"""Anomaly detected on device {alert_data['device_id']} with severity {alert_data['alert_level']}/4.
Type: {alert_data['attack_type']} | Confidence: {alert_data['confidence']:.2f}

Analyst Actions:
1. Review device logs
2. Inspect network traffic
3. Verify device integrity

Note: All actions require manual human authorization."""


def format_alert_for_dashboard(alert_data: Dict, explanation: Dict) -> str:
    """Format alert with explanation for analyst dashboard"""
    severity_names = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    output = f"""
{'='*60}
{severity_names[alert_data['alert_level']]} ALERT - IoT IDS (Local LLM)
{'='*60}
Device: {alert_data['device_id']} ({alert_data.get('device_type', 'Unknown')})
Time: {alert_data.get('timestamp', 'N/A')}
Severity: {alert_data['alert_level']*25}/100
Confidence: {alert_data['confidence']:.2f}

--- Explanation (Phi-3-mini) ---
{explanation['explanation']}

MITRE ATT&CK:
- Tactic: {explanation['mitre_tactic']}
- Technique: {explanation['mitre_technique']} ({explanation['mitre_id']})

Analyst Actions:
"""
    
    for i, action in enumerate(explanation['analyst_actions'], 1):
        output += f"{i}. {action.lstrip('0123456789.-•').strip()}\n"
    
    output += f"""
Generated in {explanation['generation_time_ms']:.0f}ms using {explanation['model']}
{'='*60}
DETECTION ONLY - No automatic mitigation applied
Human authorization required for all response actions
{'='*60}
"""
    
    return output


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("Local LLM Explanation Generator - Phi-3-mini")
    print("="*60)
    print()
    
    # Initialize
    config = LocalLLMConfig()
    generator = LocalLLMExplanationGenerator(config)
    
    # Sample alert
    sample_alert = {
        'device_id': '192.168.1.42',
        'device_type': 'IP Camera',
        'alert_level': 4,  # Critical
        'attack_type': 4,  # Exfiltration
        'confidence': 0.87,
        'timestamp': '2026-02-04 20:30:00 UTC',
        'features': {
            'dns_query_rate': 500.0,
            'dns_entropy': 0.95,
            'unique_domains': 427,
            'baseline_rate': 5.0
        }
    }
    
    # Generate explanation
    print("\nGenerating explanation...")
    explanation = generator.generate_explanation(sample_alert)
    
    # Display
    formatted = format_alert_for_dashboard(sample_alert, explanation)
    print(formatted)
    
    print(f"\nGeneration time: {explanation['generation_time_ms']:.0f}ms")
    print(f"Model: {explanation['model']}")
