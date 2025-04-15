// ========== BLOCKCHAIN COMPONENT SIMULATION CODE ==========

// Smart Contract for Voting (Solidity)
pragma solidity ^0.8.17;

contract SecureVoting {
    // Structure to represent a candidate
    struct Candidate {
        uint256 id;
        string name;
        uint256 voteCount;
    }
    
    // Structure to represent a voter
    struct Voter {
        bool isRegistered;
        bool hasVoted;
        uint256 votedCandidateId;
        bytes32 biometricHash;
    }
    
    // Election administrator address
    address public administrator;
    
    // Mapping from voter address to voter info
    mapping(address => Voter) public voters;
    
    // Array of candidates
    Candidate[] public candidates;
    
    // Election status
    bool public votingOpen;
    
    // Election name
    string public electionName;
    
    // Total votes cast
    uint256 public totalVotes;
    
    // Events
    event VoterRegistered(address voterAddress);
    event VoteCast(address voter, uint256 candidateId);
    event ElectionStarted(string name);
    event ElectionEnded(string name, uint256 totalVotes);
    
    // Modifier to restrict functions to administrator
    modifier onlyAdministrator() {
        require(msg.sender == administrator, "Only administrator can call this function");
        _;
    }
    
    // Constructor
    constructor(string memory _name) {
        administrator = msg.sender;
        electionName = _name;
        votingOpen = false;
        totalVotes = 0;
    }
    
    // Function to add a candidate
    function addCandidate(string memory _name) public onlyAdministrator {
        require(!votingOpen, "Cannot add candidate once voting has started");
        uint256 candidateId = candidates.length;
        candidates.push(Candidate(candidateId, _name, 0));
    }
    
    // Function to register a voter
    function registerVoter(address _voter, bytes32 _biometricHash) public onlyAdministrator {
        require(!voters[_voter].isRegistered, "Voter already registered");
        voters[_voter] = Voter(true, false, 0, _biometricHash);
        emit VoterRegistered(_voter);
    }
    
    // Function to start election
    function startElection() public onlyAdministrator {
        require(!votingOpen, "Election already started");
        require(candidates.length > 0, "No candidates registered");
        votingOpen = true;
        emit ElectionStarted(electionName);
    }
    
    // Function to end election
    function endElection() public onlyAdministrator {
        require(votingOpen, "Election not started yet");
        votingOpen = false;
        emit ElectionEnded(electionName, totalVotes);
    }
    
    // Function to cast vote
    function castVote(uint256 _candidateId, bytes32 _biometricVerification) public {
        Voter storage voter = voters[msg.sender];
        
        require(votingOpen, "Election is not open");
        require(voter.isRegistered, "Voter is not registered");
        require(!voter.hasVoted, "Voter has already voted");
        require(_candidateId < candidates.length, "Invalid candidate ID");
        require(voter.biometricHash == _biometricVerification, "Biometric verification failed");
        
        voter.hasVoted = true;
        voter.votedCandidateId = _candidateId;
        
        candidates[_candidateId].voteCount++;
        totalVotes++;
        
        emit VoteCast(msg.sender, _candidateId);
    }
    
    // Function to get the number of candidates
    function getCandidateCount() public view returns (uint256) {
        return candidates.length;
    }
    
    // Function to get candidate info
    function getCandidate(uint256 _candidateId) public view 
        returns (uint256 id, string memory name, uint256 voteCount) {
        require(_candidateId < candidates.length, "Invalid candidate ID");
        Candidate memory candidate = candidates[_candidateId];
        return (candidate.id, candidate.name, candidate.voteCount);
    }
    
    // Function to get election results
    function getElectionResults() public view returns (string[] memory names, uint256[] memory votes) {
        require(!votingOpen, "Election is still ongoing");
        
        names = new string[](candidates.length);
        votes = new uint256[](candidates.length);
        
        for (uint256 i = 0; i < candidates.length; i++) {
            names[i] = candidates[i].name;
            votes[i] = candidates[i].voteCount;
        }
        
        return (names, votes);
    }
}

// ========== BIOMETRIC AUTHENTICATION SIMULATION ==========

// Node.js simulation code for fingerprint authentication
const express = require('express');
const bodyParser = require('body-parser');
const crypto = require('crypto');

// Simulated fingerprint database (in production this would be securely stored)
const fingerprintDatabase = new Map();

const app = express();
app.use(bodyParser.json());

// Function to simulate fingerprint matching
function matchFingerprint(template1, template2) {
    // In a real implementation, this would use a specialized fingerprint matching algorithm
    // For simulation, we'll use a simple similarity score based on the hash proximity
    
    // Convert templates to buffers
    const buf1 = Buffer.from(template1, 'base64');
    const buf2 = Buffer.from(template2, 'base64');
    
    // Count matching bytes (simplified for simulation)
    let matchCount = 0;
    const minLength = Math.min(buf1.length, buf2.length);
    
    for (let i = 0; i < minLength; i++) {
        // Count bits that match
        const xorByte = buf1[i] ^ buf2[i];
        const matchingBits = 8 - countSetBits(xorByte);
        matchCount += matchingBits;
    }
    
    // Calculate similarity score (0-100%)
    const totalBits = minLength * 8;
    const similarityScore = (matchCount / totalBits) * 100;
    
    return similarityScore;
}

// Helper function to count set bits in a byte
function countSetBits(byte) {
    let count = 0;
    for (let i = 0; i < 8; i++) {
        if ((byte & (1 << i)) !== 0) {
            count++;
        }
    }
    return count;
}

// Endpoint to register a fingerprint
app.post('/register', (req, res) => {
    const { userId, fingerprintTemplate } = req.body;
    
    if (!userId || !fingerprintTemplate) {
        return res.status(400).json({ error: 'Missing required fields' });
    }
    
    // Store the fingerprint template
    fingerprintDatabase.set(userId, fingerprintTemplate);
    
    // In a real implementation, we would create a secure hash of the template
    const templateHash = crypto.createHash('sha256')
        .update(fingerprintTemplate)
        .digest('hex');
    
    return res.status(200).json({ 
        success: true, 
        message: 'Fingerprint registered successfully',
        templateHash
    });
});

// Endpoint to verify a fingerprint
app.post('/verify', (req, res) => {
    const { userId, fingerprintTemplate } = req.body;
    
    if (!userId || !fingerprintTemplate) {
        return res.status(400).json({ error: 'Missing required fields' });
    }
    
    // Retrieve the stored template
    const storedTemplate = fingerprintDatabase.get(userId);
    
    if (!storedTemplate) {
        return res.status(404).json({ error: 'User not registered' });
    }
    
    // Match the provided template against the stored one
    const matchScore = matchFingerprint(storedTemplate, fingerprintTemplate);
    const MATCH_THRESHOLD = 85; // 85% similarity required for a match
    
    if (matchScore >= MATCH_THRESHOLD) {
        return res.status(200).json({ 
            success: true, 
            message: 'Fingerprint verified successfully',
            matchScore
        });
    } else {
        return res.status(401).json({ 
            success: false, 
            message: 'Fingerprint verification failed',
            matchScore
        });
    }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Fingerprint authentication server running on port ${PORT}`);
});

// ========== AI SECURITY MONITORING SIMULATION ==========

// Python code for anomaly detection (fraud prevention)
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time

class VotingAnomalyDetector:
    def __init__(self, model_path=None):
        """Initialize the anomaly detector, optionally loading a pre-trained model."""
        if model_path:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(model_path.replace('.pkl', '_scaler.pkl'))
        else:
            self.model = IsolationForest(
                n_estimators=100,
                max_samples='auto',
                contamination=0.01,  # Expected proportion of anomalies
                random_state=42
            )
            self.scaler = StandardScaler()
        
        # Dictionary to track voting patterns by IP
        self.ip_tracking = {}
        # Dictionary to track voting patterns by device fingerprint
        self.device_tracking = {}
        # Track detection performance metrics
        self.metrics = {
            'total_votes': 0,
            'flagged_anomalies': 0,
            'detection_times': []
        }
    
    def train(self, training_data):
        """Train the anomaly detection model on historical voting data."""
        # Extract features from historical data
        features = self._extract_features(training_data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Train the model
        self.model.fit(scaled_features)
        
        print(f"Model trained on {len(training_data)} records")
        
        return self
    
    def save_model(self, model_path):
        """Save the trained model to disk."""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, model_path.replace('.pkl', '_scaler.pkl'))
        print(f"Model saved to {model_path}")
    
    def _extract_features(self, data):
        """Extract relevant features for anomaly detection."""
        features = pd.DataFrame()
        
        # Time-based features
        features['hour_of_day'] = data['timestamp'].dt.hour
        features['day_of_week'] = data['timestamp'].dt.dayofweek
        
        # Frequency features
        ip_counts = data['ip_address'].value_counts()
        features['ip_frequency'] = data['ip_address'].map(ip_counts)
        
        device_counts = data['device_fingerprint'].value_counts()
        features['device_frequency'] = data['device_fingerprint'].map(device_counts)
        
        # Location features
        features['location_velocity'] = data['location_velocity']  # Speed of location change
        
        # Behavioral features
        features['time_on_page'] = data['time_on_page']  # Time spent on voting page
        features['input_speed'] = data['input_speed']  # Speed of user interactions
        
        # Network features
        features['network_latency'] = data['network_latency']
        
        return features
    
    def detect_anomalies(self, vote_data):
        """
        Detect anomalies in voting behavior.
        
        Args:
            vote_data: Dictionary containing vote information
                {
                    'voter_id': str,
                    'ip_address': str,
                    'device_fingerprint': str,
                    'timestamp': datetime,
                    'location': {'lat': float, 'lng': float},
                    'time_on_page': float,
                    'input_speed': float,
                    'network_latency': float
                }
                
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        self.metrics['total_votes'] += 1
        
        # Track IP address usage
        ip = vote_data['ip_address']
        if ip in self.ip_tracking:
            self.ip_tracking[ip]['count'] += 1
            time_diff = (vote_data['timestamp'] - self.ip_tracking[ip]['last_seen']).total_seconds()
            self.ip_tracking[ip]['last_seen'] = vote_data['timestamp']
            self.ip_tracking[ip]['time_diffs'].append(time_diff)
        else:
            self.ip_tracking[ip] = {
                'count': 1,
                'last_seen': vote_data['timestamp'],
                'time_diffs': []
            }
            
        # Track device fingerprint usage
        device = vote_data['device_fingerprint']
        if device in self.device_tracking:
            self.device_tracking[device]['count'] += 1
            self.device_tracking[device]['voter_ids'].add(vote_data['voter_id'])
        else:
            self.device_tracking[device] = {
                'count': 1,
                'voter_ids': {vote_data['voter_id']}
            }
        
        # Calculate location velocity if previous location exists
        location_velocity = 0
        if 'previous_location' in vote_data:
            prev = vote_data['previous_location']
            curr = vote_data['location']
            time_diff = (vote_data['timestamp'] - vote_data['previous_timestamp']).total_seconds() / 3600  # hours
            
            # Calculate distance in kilometers using Haversine formula
            location_velocity = self._calculate_distance(prev, curr) / max(time_diff, 0.01)  # km/h
        
        # Create feature vector for this vote
        vote_features = pd.DataFrame({
            'hour_of_day': [vote_data['timestamp'].hour],
            'day_of_week': [vote_data['timestamp'].dayofweek],
            'ip_frequency': [self.ip_tracking[ip]['count']],
            'device_frequency': [self.device_tracking[device]['count']],
            'location_velocity': [location_velocity],
            'time_on_page': [vote_data['time_on_page']],
            'input_speed': [vote_data['input_speed']],
            'network_latency': [vote_data['network_latency']]
        })
        
        # Scale features
        scaled_features = self.scaler.transform(vote_features)
        
        # Predict anomaly score (-1 for anomalies, 1 for normal samples)
        score = self.model.decision_function(scaled_features)[0]
        prediction = self.model.predict(scaled_features)[0]
        
        # Apply additional rules for suspicious activity
        suspicious_flags = []
        
        # Check for multiple votes from same device
        if len(self.device_tracking[device]['voter_ids']) > 1:
            suspicious_flags.append('multiple_voters_same_device')
        
        # Check for rapid voting from same IP
        if len(self.ip_tracking[ip]['time_diffs']) > 0:
            min_time_diff = min(self.ip_tracking[ip]['time_diffs'])
            if min_time_diff < 30:  # Less than 30 seconds between votes
                suspicious_flags.append('rapid_voting_same_ip')
        
        # Check for unusual voting hours
        hour = vote_data['timestamp'].hour
        if hour < 6 or hour > 22:
            suspicious_flags.append('unusual_voting_hours')
        
        # Check for suspicious input speed (too fast might indicate automation)
        if vote_data['input_speed'] < 0.5:  # Very fast input speed
            suspicious_flags.append('suspiciously_fast_input')
            
        # Calculate detection time
        detection_time = (time.time() - start_time) * 1000  # in milliseconds
        self.metrics['detection_times'].append(detection_time)
        
        # Determine if this is an anomaly
        is_anomaly = (prediction == -1) or len(suspicious_flags) > 0
        
        if is_anomaly:
            self.metrics['flagged_anomalies'] += 1
        
        # Prepare result
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': score,
            'suspicious_flags': suspicious_flags,
            'detection_time_ms': detection_time
        }
        
        return result
    
    def _calculate_distance(self, loc1, loc2):
        """Calculate distance between two locations using Haversine formula."""
        R = 6371  # Earth radius in kilometers
        
        lat1, lon1 = np.radians(loc1['lat']), np.radians(loc1['lng'])
        lat2, lon2 = np.radians(loc2['lat']), np.radians(loc2['lng'])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_performance_metrics(self):
        """Get performance metrics of the anomaly detection system."""
        if not self.metrics['detection_times']:
            return {
                'total_votes': 0,
                'flagged_anomalies': 0,
                'avg_detection_time_ms': 0
            }
            
        return {
            'total_votes': self.metrics['total_votes'],
            'flagged_anomalies': self.metrics['flagged_anomalies'],
            'anomaly_rate': self.metrics['flagged_anomalies'] / max(self.metrics['total_votes'], 1),
            'avg_detection_time_ms': sum(self.metrics['detection_times']) / len(self.metrics['detection_times']),
            'min_detection_time_ms': min(self.metrics['detection_times']),
            'max_detection_time_ms': max(self.metrics['detection_times'])
        }

# Example usage of the anomaly detector in simulation
if __name__ == "__main__":
    # Load sample historical voting data
    historical_data = pd.read_csv('historical_voting_data.csv')
    historical_data['timestamp'] = pd.to_datetime(historical_data['timestamp'])
    
    # Create and train the model
    detector = VotingAnomalyDetector()
    detector.train(historical_data)
    
    # Save the model
    detector.save_model('voting_anomaly_detector.pkl')
    
    # Simulate real-time votes
    import datetime
    
    # Normal vote
    normal_vote = {
        'voter_id': '1234567890',
        'ip_address': '192.168.1.100',
        'device_fingerprint': 'abcdef123456',
        'timestamp': datetime.datetime.now(),
        'location': {'lat': 37.7749, 'lng': -122.4194},
        'time_on_page': 45.3,  # seconds
        'input_speed': 2.1,    # actions per second
        'network_latency': 120  # milliseconds
    }
    
    # Suspicious vote (rapid voting from same IP)
    suspicious_vote = {
        'voter_id': '0987654321',
        'ip_address': '192.168.1.100',  # Same IP as normal vote
        'device_fingerprint': 'xyz789012',
        'timestamp': datetime.datetime.now() + datetime.timedelta(seconds=5),  # Very soon after
        'location': {'lat': 37.7749, 'lng': -122.4194},
        'time_on_page': 3.1,   # Very short time on page
        'input_speed': 10.5,   # Very fast input (suspicious)
        'network_latency': 118  # milliseconds
    }
    
    # Detect anomalies
    normal_result = detector.detect_anomalies(normal_vote)
    suspicious_result = detector.detect_anomalies(suspicious_vote)
    
    print("Normal vote detection result:", normal_result)
    print("Suspicious vote detection result:", suspicious_result)
    print("Detector performance metrics:", detector.get_performance_metrics())
'''

// ========== LOAD TESTING SIMULATION ==========

// JMeter Test Plan for Load Testing (in XML format)
'''
<?xml version="1.0" encoding="UTF-8"?>
<jmeterTestPlan version="1.2" properties="5.0" jmeter="5.5">
  <hashTree>
    <TestPlan guiclass="TestPlanGui" testclass="TestPlan" testname="Secure Voting System Load Test" enabled="true">
      <stringProp name="TestPlan.comments">Load Test for Blockchain-Based Voting System</stringProp>
      <boolProp name="TestPlan.functional_mode">false</boolProp>
      <boolProp name="TestPlan.tearDown_on_shutdown">true</boolProp>
      <boolProp name="TestPlan.serialize_threadgroups">false</boolProp>
      <elementProp name="TestPlan.user_defined_variables" elementType="Arguments" guiclass="ArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
        <collectionProp name="Arguments.arguments"/>
      </elementProp>
      <stringProp name="TestPlan.user_define_classpath"></stringProp>
    </TestPlan>
    <hashTree>
      <ThreadGroup guiclass="ThreadGroupGui" testclass="ThreadGroup" testname="Voter Simulation" enabled="true">
        <stringProp name="ThreadGroup.on_sample_error">continue</stringProp>
        <elementProp name="ThreadGroup.main_controller" elementType="LoopController" guiclass="LoopControlPanel" testclass="LoopController" testname="Loop Controller" enabled="true">
          <boolProp name="LoopController.continue_forever">false</boolProp>
          <stringProp name="LoopController.loops">1</stringProp>
        </elementProp>
        <stringProp name="ThreadGroup.num_threads">1000</stringProp>
        <stringProp name="ThreadGroup.ramp_time">60</stringProp>
        <boolProp name="ThreadGroup.scheduler">false</boolProp>
        <stringProp name="ThreadGroup.duration"></stringProp>
        <stringProp name="ThreadGroup.delay"></stringProp>
        <boolProp name="ThreadGroup.same_user_on_next_iteration">false</boolProp>
      </ThreadGroup>
      <hashTree>
        <CSVDataSet guiclass="TestBeanGUI" testclass="CSVDataSet" testname="Voter Data" enabled="true">
          <stringProp name="delimiter">,</stringProp>
          <stringProp name="fileEncoding">UTF-8</stringProp>
          <stringProp name="filename">voter_data.csv</stringProp>
          <boolProp name="ignoreFirstLine">true</boolProp>
          <boolProp name="quotedData">false</boolProp>
          <boolProp name="recycle">true</boolProp>
          <stringProp name="shareMode">shareMode.all</stringProp>
          <boolProp name="stopThread">false</boolProp>
          <stringProp name="variableNames">voter_id,fingerprint_id,face_id</stringProp>
        </CSVDataSet>
        <hashTree/>
        <ConfigTestElement guiclass="HttpDefaultsGui" testclass="ConfigTestElement" testname="HTTP Request Defaults" enabled="true">
          <elementProp name="HTTPsampler.Arguments" elementType="Arguments" guiclass="HTTPArgumentsPanel" testclass="Arguments" testname="User Defined Variables" enabled="true">
            <collectionProp name="Arguments.arguments"/>
          </elementProp>
          <stringProp name="HTTPSampler.domain">voting-system-api.example.com</stringProp>
          <stringProp name="HTTPSampler.port">443</stringProp>
          <stringProp name="HTTPSampler.protocol">https</stringProp>
          <stringProp name="HTTPSampler.contentEncoding"></stringProp>
          <stringProp name="HTTPSampler.path"></stringProp>
          <stringProp name="HTTPSampler.concurrentPool">6</stringProp>
          <stringProp name="HTTPSampler.connect_timeout">5000</stringProp>
          <stringProp name="HTTPSampler.response_timeout">30000</stringProp>
        </ConfigTestElement>
        <hashTree/>
        <HeaderManager guiclass="HeaderPanel" testclass="HeaderManager" testname="HTTP Header Manager" enabled="true">
          <collectionProp name="HeaderManager.headers">
            <elementProp name="" elementType="Header">
              <stringProp name="Header.name">Content-Type</stringProp>
              <stringProp name="Header.value">application/json</stringProp>
            </elementProp>
            <elementProp name="" elementType="Header">
              <stringProp name="Header.name">Accept</stringProp>
              <stringProp name="Header.value">application/json</stringProp>
            </elementProp>
          </collectionProp>
        </HeaderManager>
        <hashTree/>
        <TransactionController guiclass="TransactionControllerGui" testclass="TransactionController" testname="Complete Voting Process" enabled="true">
          <boolProp name="TransactionController.includeTimers">false</boolProp>
          <boolProp name="TransactionController.parent">true</boolProp>
        </TransactionController>
        <hashTree>
          <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="Login Request" enabled="true">
            <boolProp name="HTTPSampler.postBodyRaw">true</boolProp>
            <elementProp name="HTTPsampler.Arguments" elementType="Arguments">
              <collectionProp name="Arguments.arguments">
                <elementProp name="" elementType="HTTPArgument">
                  <boolProp name="HTTPArgument.always_encode">false</boolProp>
                  <stringProp name="Argument.value">{
  "voter_id": "${voter_id}"
}</stringProp>
                  <stringProp name="Argument.metadata">=</stringProp>
                </elementProp>
              </collectionProp>
            </elementProp>
            <stringProp name="HTTPSampler.domain"></stringProp>
            <stringProp name="HTTPSampler.port"></stringProp>
            <stringProp name="HTTPSampler.protocol"></stringProp>
            <stringProp name="HTTPSampler.contentEncoding"></stringProp>
            <stringProp name="HTTPSampler.path">/api/auth/login</stringProp>
            <stringProp name="HTTPSampler.method">POST</stringProp>
            <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
            <boolProp name="HTTPSampler.auto_redirects">false</boolProp>
            <boolProp name="HTTPSampler.use_keepalive">true</boolProp>
            <boolProp name="HTTPSampler.DO_MULTIPART_POST">false</boolProp>
            <stringProp name="HTTPSampler.embedded_url_re"></stringProp>
            <stringProp name="HTTPSampler.connect_timeout"></stringProp>
            <stringProp name="HTTPSampler.response_timeout"></stringProp>
          </HTTPSamplerProxy>
          <hashTree>
            <JSONPostProcessor guiclass="JSONPostProcessorGui" testclass="JSONPostProcessor" testname="Extract Session Token" enabled="true">
              <stringProp name="JSONPostProcessor.referenceNames">session_token</stringProp>
              <stringProp name="JSONPostProcessor.jsonPathExprs">$.session_token</stringProp>
              <stringProp name="JSONPostProcessor.match_numbers"></stringProp>
            </JSONPostProcessor>
            <hashTree/>
            <JSONPostProcessor guiclass="JSONPostProcessorGui" testclass="JSONPostProcessor" testname="Extract OTP" enabled="true">
              <stringProp name="JSONPostProcessor.referenceNames">otp</stringProp>
              <stringProp name="JSONPostProcessor.jsonPathExprs">$.otp</stringProp>
              <stringProp name="JSONPostProcessor.match_numbers"></stringProp>
            </JSONPostProcessor>
            <hashTree/>
          </hashTree>
          <TestAction guiclass="TestActionGui" testclass="TestAction" testname="Think Time" enabled="true">
            <intProp name="ActionProcessor.action">1</intProp>
            <intProp name="ActionProcessor.target">0</intProp>
            <stringProp name="ActionProcessor.duration">1000</stringProp>
          </TestAction>
          <hashTree/>
          <HTTPSamplerProxy guiclass="HttpTestSampleGui" testclass="HTTPSamplerProxy" testname="OTP Verification" enabled="true">
            <boolProp name="HTTPSampler.postBodyRaw">true</boolProp>
            <elementProp name="HTTPsampler.Arguments" elementType="Arguments">
              <collectionProp name="Arguments.arguments">
                <elementProp name="" elementType="HTTPArgument">
                  <boolProp name="HTTPArgument.always_encode">false</boolProp>
                  <stringProp name="Argument.value">{
  "voter_id": "${voter_id}",
  "otp": "${otp}",
  "session_token": "${session_token}"
}</stringProp>
                  <stringProp name="Argument.metadata">=</stringProp>
                </elementProp>
              </collectionProp>
            </elementProp>
            <stringProp name="HTTPSampler.domain"></stringProp>
            <stringProp name="HTTPSampler.port"></stringProp>
            <stringProp name="HTTPSampler.protocol"></stringProp>
            <stringProp name="HTTPSampler.contentEncoding"></stringProp>
            <stringProp name="HTTPSampler.path">/api/auth/verify-otp</stringProp>
            <stringProp name="HTTPSampler.method">POST</stringProp>
            <boolProp name="HTTPSampler.follow_redirects">true</boolProp>
            <boolProp name="HTTPSampler.auto_redirects">false