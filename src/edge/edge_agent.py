"""
Edge Agent: Lightweight IDS agent for IoT devices
Performs semantic encoding and local anomaly detection

Components:
- MiniLM semantic encoder (22M params)
- Statistical feature extractor
- Local anomaly detector
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from sentence_transformers import SentenceTransformer
from dataclasses import dataclass
import time


@dataclass
class EdgeConfig:
    """Configuration for edge agent"""
    device_id: str
    model_name: str = 'all-MiniLM-L6-v2'
    embedding_dim: int = 384
    statistical_feature_dim: int = 50
    max_latency_ms: float = 50.0
    memory_limit_mb: int = 512
    

class SemanticEncoder:
    """
    Lightweight LLM-based semantic encoder for traffic analysis
    Uses MiniLM for efficient edge deployment
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize semantic encoder
        
        Args:
            model_name: HuggingFace model name (default: MiniLM-L6)
        """
        self.model = SentenceTransformer(model_name)
        self.model.eval()
        
        # Quantize for edge deployment (optional)
        # self.model = torch.quantization.quantize_dynamic(
        #     self.model, {torch.nn.Linear}, dtype=torch.qint8
        # )
        
    def encode_traffic(self, traffic_snippet: str) -> np.ndarray:
        """
        Encode traffic behavior into semantic embedding
        
        Args:
            traffic_snippet: Protocol sequence (e.g., "MQTT CONNECT → PUBLISH")
            
        Returns:
            384-dim embedding vector
        """
        with torch.no_grad():
            embedding = self.model.encode(
                traffic_snippet, 
                convert_to_tensor=False,
                show_progress_bar=False
            )
        return embedding
    
    def encode_batch(self, traffic_snippets: List[str]) -> np.ndarray:
        """Batch encoding for efficiency"""
        with torch.no_grad():
            embeddings = self.model.encode(
                traffic_snippets,
                convert_to_tensor=False,
                show_progress_bar=False,
                batch_size=32
            )
        return embeddings


class StatisticalFeatureExtractor:
    """
    Extract statistical features from network traffic
    Output: 50-dimensional feature vector
    """
    
    def extract(self, packets: List[Dict]) -> np.ndarray:
        """
        Extract features from packet list
        
        Args:
            packets: List of packet dictionaries with keys:
                - timestamp, size, protocol, src_port, dst_port, flags
                
        Returns:
            50-dim statistical feature vector
        """
        features = {}
        
        # Basic statistics (10 features)
        features['packet_count'] = len(packets)
        features['total_bytes'] = sum(p.get('size', 0) for p in packets)
        features['avg_packet_size'] = features['total_bytes'] / max(len(packets), 1)
        
        # Timing features (10 features)
        timestamps = [p.get('timestamp', 0) for p in packets]
        if len(timestamps) > 1:
            inter_arrival_times = np.diff(timestamps)
            features['iat_mean'] = np.mean(inter_arrival_times)
            features['iat_std'] = np.std(inter_arrival_times)
            features['iat_median'] = np.median(inter_arrival_times)
            features['flow_duration'] = timestamps[-1] - timestamps[0]
        else:
            features['iat_mean'] = 0
            features['iat_std'] = 0
            features['iat_median'] = 0
            features['flow_duration'] = 0
            
        # Protocol distribution (10 features)
        protocols = [p.get('protocol', 'unknown') for p in packets]
        protocol_counts = {}
        for proto in ['TCP', 'UDP', 'MQTT', 'CoAP', 'HTTP', 'HTTPS']:
            protocol_counts[proto] = protocols.count(proto) / max(len(packets), 1)
        features.update(protocol_counts)
        
        # Port statistics (10 features)
        src_ports = [p.get('src_port', 0) for p in packets]
        dst_ports = [p.get('dst_port', 0) for p in packets]
        features['unique_src_ports'] = len(set(src_ports))
        features['unique_dst_ports'] = len(set(dst_ports))
        features['src_port_entropy'] = self._entropy(src_ports)
        features['dst_port_entropy'] = self._entropy(dst_ports)
        
        # Packet size distribution (10 features)
        sizes = [p.get('size', 0) for p in packets]
        features['size_mean'] = np.mean(sizes)
        features['size_std'] = np.std(sizes)
        features['size_median'] = np.median(sizes)
        features['size_95percentile'] = np.percentile(sizes, 95) if sizes else 0
        
        # Convert to feature vector (pad/truncate to 50 dims)
        feature_vector = np.zeros(50)
        feature_list = list(features.values())[:50]
        feature_vector[:len(feature_list)] = feature_list
        
        return feature_vector
    
    @staticmethod
    def _entropy(values: List) -> float:
        """Calculate Shannon entropy"""
        if not values:
            return 0.0
        value_counts = {}
        for v in values:
            value_counts[v] = value_counts.get(v, 0) + 1
        total = len(values)
        entropy = -sum(
            (count/total) * np.log2(count/total) 
            for count in value_counts.values()
        )
        return entropy


class LocalAnomalyDetector:
    """
    Simple local anomaly detection using statistical thresholds
    Not used for final decision, only preliminary scoring
    """
    
    def __init__(self, threshold: float = 2.5):
        """
        Args:
            threshold: Number of standard deviations for anomaly
        """
        self.threshold = threshold
        self.baseline_mean = None
        self.baseline_std = None
        
    def fit(self, normal_features: np.ndarray):
        """Fit on normal traffic baseline"""
        self.baseline_mean = np.mean(normal_features, axis=0)
        self.baseline_std = np.std(normal_features, axis=0) + 1e-6
        
    def score(self, features: np.ndarray) -> float:
        """
        Compute anomaly score (0-1)
        
        Returns:
            Anomaly score (higher = more anomalous)
        """
        if self.baseline_mean is None:
            return 0.5  # Uncalibrated
            
        # Z-score based anomaly
        z_scores = np.abs((features - self.baseline_mean) / self.baseline_std)
        max_z = np.max(z_scores)
        
        # Convert to 0-1 score
        anomaly_score = min(max_z / (self.threshold * 2), 1.0)
        return anomaly_score


class EdgeAgent:
    """
    Main edge agent integrating all components
    
    Responsibilities:
    - Capture and process network traffic
    - Extract statistical + semantic features
    - Compute local anomaly score
    - Send features to Fog layer
    
    Constraints:
    - <512MB memory
    - <50ms latency per sample
    - No decision authority (only feature extraction)
    """
    
    def __init__(self, config: EdgeConfig):
        self.config = config
        self.semantic_encoder = SemanticEncoder(config.model_name)
        self.feature_extractor = StatisticalFeatureExtractor()
        self.anomaly_detector = LocalAnomalyDetector()
        
        print(f"Edge Agent {config.device_id} initialized")
        print(f"  - Semantic encoder: {config.model_name}")
        print(f"  - Embedding dim: {config.embedding_dim}")
        print(f"  - Memory limit: {config.memory_limit_mb}MB")
        print(f"  - Max latency: {config.max_latency_ms}ms")
        
    def process_traffic(self, packets: List[Dict]) -> Dict:
        """
        Process traffic sample and extract features
        
        Args:
            packets: List of packet dictionaries
            
        Returns:
            Dictionary with:
                - device_id
                - features (statistical + semantic)
                - local_anomaly_score
                - timestamp
                - latency_ms
        """
        start_time = time.time()
        
        # Extract statistical features
        statistical_features = self.feature_extractor.extract(packets)
        
        # Generate traffic snippet for semantic encoding
        traffic_snippet = self._format_traffic_snippet(packets)
        semantic_embedding = self.semantic_encoder.encode_traffic(traffic_snippet)
        
        # Concatenate features
        full_features = np.concatenate([statistical_features, semantic_embedding])
        
        # Local anomaly score
        local_anomaly_score = self.anomaly_detector.score(statistical_features)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Verify latency constraint
        if latency_ms > self.config.max_latency_ms:
            print(f"WARNING: Latency {latency_ms:.2f}ms exceeds limit "
                  f"{self.config.max_latency_ms}ms")
        
        return {
            'device_id': self.config.device_id,
            'features': full_features,
            'local_anomaly_score': local_anomaly_score,
            'timestamp': time.time(),
            'latency_ms': latency_ms,
            'packet_count': len(packets)
        }
    
    @staticmethod
    def _format_traffic_snippet(packets: List[Dict]) -> str:
        """
        Format packets into human-readable traffic snippet
        Example: "MQTT CONNECT → MQTT CONNACK → MQTT PUBLISH (topic: /sensor/temp)"
        """
        if not packets:
            return "Empty traffic"
            
        protocols = [p.get('protocol', 'UNKNOWN') for p in packets[:10]]  # First 10
        snippet = " → ".join(protocols)
        
        # Add context if available
        if packets[0].get('mqtt_topic'):
            snippet += f" (topic: {packets[0]['mqtt_topic']})"
            
        return snippet
    
    def calibrate(self, normal_traffic_samples: List[List[Dict]]):
        """
        Calibrate anomaly detector on normal traffic baseline
        
        Args:
            normal_traffic_samples: List of normal traffic samples (list of packets)
        """
        print(f"Calibrating edge agent {self.config.device_id}...")
        
        normal_features = []
        for packets in normal_traffic_samples:
            features = self.feature_extractor.extract(packets)
            normal_features.append(features)
            
        normal_features = np.array(normal_features)
        self.anomaly_detector.fit(normal_features)
        
        print(f"Calibration complete: {len(normal_features)} samples")


# Example usage
if __name__ == "__main__":
    # Initialize edge agent
    config = EdgeConfig(device_id="edge_001")
    agent = EdgeAgent(config)
    
    # Simulate traffic sample
    sample_packets = [
        {'timestamp': 0.0, 'size': 128, 'protocol': 'MQTT', 
         'src_port': 1883, 'dst_port': 12345, 'mqtt_topic': '/sensor/temp'},
        {'timestamp': 0.1, 'size': 64, 'protocol': 'MQTT', 
         'src_port': 12345, 'dst_port': 1883},
        {'timestamp': 0.2, 'size': 256, 'protocol': 'MQTT', 
         'src_port': 1883, 'dst_port': 12345, 'mqtt_topic': '/sensor/humidity'},
    ]
    
    # Process traffic
    result = agent.process_traffic(sample_packets)
    
    print("\nProcessing result:")
    print(f"  Device: {result['device_id']}")
    print(f"  Feature dim: {result['features'].shape}")
    print(f"  Anomaly score: {result['local_anomaly_score']:.3f}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")
    print(f"  Packets: {result['packet_count']}")
