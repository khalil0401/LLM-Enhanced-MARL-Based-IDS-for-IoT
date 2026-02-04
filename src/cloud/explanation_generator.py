"""
Cloud Layer: LLM-based Explanation Generator
Provides human-interpretable explanations for IDS alerts
"""

import openai
from typing import Dict, Optional, List
from dataclasses import dataclass
import json
import time


@dataclass
class ExplanationConfig:
    """Configuration for explanation generator"""
    model: str = "gpt-4"
    temperature: float = 0.3  # Lower = more deterministic
    max_tokens: int = 300
    api_key: Optional[str] = None


class MITREMapper:
    """
    Maps attack types to MITRE ATT&CK for IoT taxonomy
    """
    
    def __init__(self):
        # Simplified mapping (expand with full MITRE ATT&CK for IoT)
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
        """
        Map attack type ID to MITRE ATT&CK entry
        
        Args:
            attack_type: Integer attack type (0-6)
            
        Returns:
            Dictionary with id, tactic, technique
        """
        return self.attack_mapping.get(attack_type, self.attack_mapping[0])


class ExplanationGenerator:
    """
    GPT-4 based explanation generator for IDS alerts
    
    Safety constraints:
    - Explanations are DESCRIPTIVE only (what was detected)
    - NO prescriptive actions (no "system will block")
    - All recommendations require human authorization
    """
    
    def __init__(self, config: ExplanationConfig):
        self.config = config
        
        # Set API key
        if config.api_key:
            openai.api_key = config.api_key
        
        self.mitre_mapper = MITREMapper()
        
        print(f"Explanation Generator initialized:")
        print(f"  Model: {config.model}")
        print(f"  Temperature: {config.temperature}")
        print(f"  Max tokens: {config.max_tokens}")
    
    def generate_explanation(self, alert_data: Dict) -> Dict[str, str]:
        """
        Generate human-readable explanation for alert
        
        Args:
            alert_data: Dictionary with:
                - device_id: Device identifier
                - device_type: Type of IoT device
                - alert_level: 0-4 severity
                - attack_type: 0-6 attack category
                - confidence: 0-1 confidence score
                - features: Feature dictionary or summary
                - timestamp: Alert timestamp
                
        Returns:
            Dictionary with:
                - explanation: Main explanation text
                - mitre_tactic: ATT&CK tactic
                - mitre_technique: ATT&CK technique
                - analyst_actions: Suggested investigative actions
                - confidence_justification: Why this confidence score
        """
        start_time = time.time()
        
        # Map to MITRE ATT&CK
        mitre_info = self.mitre_mapper.map_attack(alert_data['attack_type'])
        
        # Construct prompt
        prompt = self._build_prompt(alert_data, mitre_info)
        
        # Call GPT-4
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            explanation_text = response['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            explanation_text = self._get_fallback_explanation(alert_data)
        
        # Parse structured output
        result = self._parse_explanation(explanation_text, mitre_info)
        
        # Add metadata
        result['generation_time_ms'] = (time.time() - start_time) * 1000
        result['model'] = self.config.model
        
        return result
    
    def _get_system_prompt(self) -> str:
        """System prompt defining LLM role and constraints"""
        return """You are a cybersecurity analyst assistant specializing in IoT security. 
Your role is to explain intrusion detection alerts to human analysts.

CRITICAL CONSTRAINTS:
- Be DESCRIPTIVE: Explain what was detected and why
- Be FACTUAL: Based only on provided evidence
- NO PRESCRIPTION: Never suggest automated blocking, quarantine, or mitigation
- HUMAN AUTHORITY: All actions require explicit human authorization
- NO SPECULATION: If uncertain, say so clearly

Output format:
1. Behavior Summary (2-3 sentences)
2. MITRE ATT&CK Context (tactic and technique)
3. Key Anomalies (specific feature deviations)
4. Confidence Justification (why this score)
5. Analyst Actions (observational only, manual execution)
"""
    
    def _build_prompt(self, alert_data: Dict, mitre_info: Dict) -> str:
        """Build user prompt with alert details"""
        
        attack_type_names = [
            "None", "DDoS", "Malware", "Reconnaissance", 
            "Data Exfiltration", "Command Injection", "Firmware Manipulation"
        ]
        
        prompt = f"""
IoT Network Alert Analysis:

Device Information:
- Device ID: {alert_data['device_id']}
- Device Type: {alert_data.get('device_type', 'Unknown IoT Device')}
- Timestamp: {alert_data.get('timestamp', 'N/A')}

Alert Details:
- Severity: {alert_data['alert_level']}/4 ({['Info', 'Low', 'Medium', 'High', 'Critical'][alert_data['alert_level']]})
- Attack Type: {attack_type_names[alert_data['attack_type']]}
- Confidence: {alert_data['confidence']:.2f}

MITRE ATT&CK Mapping:
- Technique ID: {mitre_info['id']}
- Tactic: {mitre_info['tactic']}
- Technique: {mitre_info['technique']}

Detected Anomalies:
{self._format_features(alert_data.get('features', {}))}

Task: Provide a concise explanation of this alert following the required format.
Remember: Be descriptive, not prescriptive. All actions require human authorization.
"""
        return prompt
    
    def _format_features(self, features: Dict) -> str:
        """Format feature anomalies for prompt"""
        if not features:
            return "- No specific feature data available"
        
        # Extract top anomalous features (simplified)
        feature_lines = []
        for key, value in list(features.items())[:5]:  # Top 5
            feature_lines.append(f"- {key}: {value}")
        
        return "\n".join(feature_lines)
    
    def _parse_explanation(self, text: str, mitre_info: Dict) -> Dict[str, str]:
        """
        Parse LLM output into structured format
        
        Returns:
            Dictionary with explanation components
        """
        # Simple parsing (could be improved with structured output)
        return {
            'explanation': text,
            'mitre_tactic': mitre_info['tactic'],
            'mitre_technique': mitre_info['technique'],
            'mitre_id': mitre_info['id'],
            'analyst_actions': self._extract_actions(text),
            'confidence_justification': self._extract_justification(text)
        }
    
    def _extract_actions(self, text: str) -> List[str]:
        """Extract analyst action recommendations from text"""
        # Simple extraction (look for numbered lists after "Analyst Actions")
        lines = text.split('\n')
        actions = []
        in_actions_section = False
        
        for line in lines:
            if 'Analyst Actions' in line or 'Recommended Actions' in line:
                in_actions_section = True
                continue
            
            if in_actions_section and line.strip().startswith(('1.', '2.', '3.', '4.', '5.', '-')):
                actions.append(line.strip())
        
        return actions if actions else ["Review alert details and device logs"]
    
    def _extract_justification(self, text: str) -> str:
        """Extract confidence justification from text"""
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if 'Confidence' in line and i+1 < len(lines):
                return lines[i+1].strip()
        return "Based on observed anomalies"
    
    def _get_fallback_explanation(self, alert_data: Dict) -> str:
        """Fallback explanation if LLM fails"""
        return f"""
An anomaly was detected on device {alert_data['device_id']} with severity {alert_data['alert_level']}/4.
The system detected behavior consistent with {alert_data['attack_type']} activity.
Confidence score: {alert_data['confidence']:.2f}.

Analyst Actions:
1. Review device logs for suspicious activity
2. Inspect network traffic patterns
3. Verify device firmware integrity
4. Consider forensic analysis if warranted

Note: This is a fallback explanation due to LLM unavailability.
All actions require manual human authorization.
"""


def format_alert_for_dashboard(alert_data: Dict, explanation: Dict) -> str:
    """
    Format complete alert with explanation for analyst dashboard
    
    Args:
        alert_data: Original alert data
        explanation: Generated explanation
        
    Returns:
        Formatted text for display
    """
    severity_names = ['INFO', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    output = f"""
{'='*60}
{severity_names[alert_data['alert_level']]} ALERT - IoT IDS
{'='*60}
Device: {alert_data['device_id']} ({alert_data.get('device_type', 'Unknown')})
Timestamp: {alert_data.get('timestamp', 'N/A')}
Severity: {alert_data['alert_level']*25}/100
Confidence: {alert_data['confidence']:.2f}

--- Explanation ---
{explanation['explanation']}

MITRE ATT&CK Mapping:
- Tactic: {explanation['mitre_tactic']}
- Technique: {explanation['mitre_technique']} ({explanation['mitre_id']})

Suggested Analyst Actions:
"""
    
    for i, action in enumerate(explanation['analyst_actions'], 1):
        output += f"{i}. {action.lstrip('0123456789.-').strip()}\n"
    
    output += f"""
{'='*60}
Note: This is a DETECTION-ONLY alert. No automatic mitigation
has been applied. Analyst authorization required for any
response actions.
{'='*60}
"""
    
    return output


# Example usage
if __name__ == "__main__":
    # Initialize (requires OpenAI API key)
    config = ExplanationConfig(
        model="gpt-4",
        api_key="your-api-key-here"  # Replace with actual key
    )
    
    generator = ExplanationGenerator(config)
    
    # Sample alert
    sample_alert = {
        'device_id': '192.168.1.42',
        'device_type': 'Hikvision IP Camera',
        'alert_level': 4,  # Critical
        'attack_type': 4,  # Data Exfiltration
        'confidence': 0.87,
        'timestamp': '2026-02-04 18:30:15 UTC',
        'features': {
            'dns_query_rate': 500.0,
            'dns_query_entropy': 0.95,
            'unique_domains': 427,
            'baseline_dns_rate': 5.0
        }
    }
    
    # Generate explanation
    print("Generating explanation...")
    explanation = generator.generate_explanation(sample_alert)
    
    # Format for display
    formatted = format_alert_for_dashboard(sample_alert, explanation)
    print(formatted)
    
    print(f"\nGeneration time: {explanation['generation_time_ms']:.2f}ms")
