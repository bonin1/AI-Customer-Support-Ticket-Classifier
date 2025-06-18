"""
Audit and Compliance System
Provides comprehensive logging, audit trails, and compliance reporting for ML models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import json
import os
import sqlite3
import logging
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import uuid
import warnings
warnings.filterwarnings('ignore')

@dataclass
class AuditEvent:
    """Data class for audit events."""
    event_id: str
    timestamp: datetime
    event_type: str  # prediction, training, update, access, etc.
    user_id: Optional[str]
    model_id: str
    input_data_hash: str
    output_data: Dict[str, Any]
    metadata: Dict[str, Any]
    compliance_tags: List[str]
    data_lineage: Dict[str, Any]

@dataclass
class ComplianceRule:
    """Data class for compliance rules."""
    rule_id: str
    name: str
    description: str
    rule_type: str  # retention, access, bias, fairness, etc.
    parameters: Dict[str, Any]
    enabled: bool = True
    severity: str = "medium"  # low, medium, high, critical

class MLAuditSystem:
    """
    Comprehensive ML audit and compliance system that:
    - Logs all model interactions and decisions
    - Tracks data lineage and model provenance
    - Monitors for bias and fairness issues
    - Ensures compliance with regulations (GDPR, CCPA, etc.)
    - Provides audit trails and reports
    - Implements data retention policies
    """
    
    def __init__(self, 
                 db_path: str = "models/audit.db",
                 retention_days: int = 365,
                 enable_gdpr: bool = True,
                 enable_bias_monitoring: bool = True):
        """
        Initialize audit system.
        
        Args:
            db_path: Path to audit database
            retention_days: Data retention period in days
            enable_gdpr: Enable GDPR compliance features
            enable_bias_monitoring: Enable bias monitoring
        """
        self.db_path = db_path
        self.retention_days = retention_days
        self.enable_gdpr = enable_gdpr
        self.enable_bias_monitoring = enable_bias_monitoring
        
        # Compliance rules
        self.compliance_rules = {}
        
        # Audit statistics
        self.audit_stats = {
            'total_events': 0,
            'events_by_type': defaultdict(int),
            'compliance_violations': 0,
            'gdpr_requests': 0
        }
        
        # Setup database and logging
        self._setup_database()
        self._setup_default_rules()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _setup_database(self):
        """Setup audit database."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
          # Audit events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                user_id TEXT,
                model_id TEXT NOT NULL,
                input_data_hash TEXT,
                output_data TEXT,
                metadata TEXT,
                compliance_tags TEXT,
                data_lineage TEXT
            )
        ''')
        
        # Create indexes for audit_events
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_user_id ON audit_events(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_audit_model_id ON audit_events(model_id)')
          # Compliance violations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS compliance_violations (
                violation_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                event_id TEXT,
                severity TEXT,
                description TEXT,
                resolved BOOLEAN DEFAULT FALSE,
                resolution_notes TEXT
            )
        ''')
        
        # Create indexes for compliance_violations
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_timestamp ON compliance_violations(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_rule_id ON compliance_violations(rule_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_violations_severity ON compliance_violations(severity)')
        
        # Data lineage table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_lineage (
                lineage_id TEXT PRIMARY KEY,
                data_hash TEXT NOT NULL,
                source_type TEXT,
                source_info TEXT,
                transformations TEXT,
                timestamp TEXT
            )
        ''')
        
        # Create indexes for data_lineage
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_data_hash ON data_lineage(data_hash)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_lineage_timestamp ON data_lineage(timestamp)')
        
        # GDPR requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS gdpr_requests (
                request_id TEXT PRIMARY KEY,
                request_type TEXT NOT NULL,
                user_id TEXT,
                email TEXT,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                processed_timestamp TEXT,
                notes TEXT
            )
        ''')
        
        # Create indexes for gdpr_requests
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gdpr_user_id ON gdpr_requests(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gdpr_email ON gdpr_requests(email)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_gdpr_timestamp ON gdpr_requests(timestamp)')
        
        conn.commit()
        conn.close()
    
    def _setup_default_rules(self):
        """Setup default compliance rules."""
        default_rules = [
            ComplianceRule(
                rule_id="data_retention",
                name="Data Retention Policy",
                description=f"Delete audit data older than {self.retention_days} days",
                rule_type="retention",
                parameters={"retention_days": self.retention_days},
                severity="high"
            ),
            ComplianceRule(
                rule_id="bias_monitoring",
                name="Model Bias Monitoring",
                description="Monitor for potential bias in model predictions",
                rule_type="bias",
                parameters={"threshold": 0.1, "protected_attributes": []},
                enabled=self.enable_bias_monitoring,
                severity="high"
            ),
            ComplianceRule(
                rule_id="access_logging",
                name="Access Logging",
                description="Log all model access events",
                rule_type="access",
                parameters={},
                severity="medium"
            )
        ]
        
        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule
    
    def log_prediction(self, 
                      model_id: str,
                      input_text: str,
                      prediction: str,
                      confidence: float,
                      user_id: Optional[str] = None,
                      metadata: Dict[str, Any] = None) -> str:
        """
        Log a model prediction event.
        
        Args:
            model_id: Model identifier
            input_text: Input text
            prediction: Model prediction
            confidence: Prediction confidence
            user_id: Optional user identifier
            metadata: Additional metadata
            
        Returns:
            Event ID
        """
        event_id = str(uuid.uuid4())
        input_hash = self._hash_data(input_text)
        
        # Prepare compliance tags
        compliance_tags = ["prediction"]
        if self.enable_gdpr:
            compliance_tags.append("gdpr_relevant")
        
        # Create audit event
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type="prediction",
            user_id=user_id,
            model_id=model_id,
            input_data_hash=input_hash,
            output_data={
                "prediction": prediction,
                "confidence": confidence
            },
            metadata=metadata or {},
            compliance_tags=compliance_tags,
            data_lineage=self._create_data_lineage(input_text, "user_input")
        )
        
        # Store event
        self._store_audit_event(event)
        
        # Check compliance
        self._check_compliance(event)
        
        return event_id
    
    def log_training_event(self,
                          model_id: str,
                          dataset_info: Dict[str, Any],
                          training_params: Dict[str, Any],
                          results: Dict[str, Any],
                          user_id: Optional[str] = None) -> str:
        """Log a model training event."""
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type="training",
            user_id=user_id,
            model_id=model_id,
            input_data_hash=self._hash_data(str(dataset_info)),
            output_data={
                "training_params": training_params,
                "results": results
            },
            metadata={"dataset_info": dataset_info},
            compliance_tags=["training", "model_creation"],
            data_lineage=self._create_data_lineage(dataset_info, "training_data")
        )
        
        self._store_audit_event(event)
        self._check_compliance(event)
        
        return event_id
    
    def log_model_update(self,
                        model_id: str,
                        update_type: str,
                        changes: Dict[str, Any],
                        user_id: Optional[str] = None) -> str:
        """Log a model update event."""
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type="model_update",
            user_id=user_id,
            model_id=model_id,
            input_data_hash=self._hash_data(str(changes)),
            output_data={
                "update_type": update_type,
                "changes": changes
            },
            metadata={},
            compliance_tags=["model_modification"],
            data_lineage={}
        )
        
        self._store_audit_event(event)
        self._check_compliance(event)
        
        return event_id
    
    def log_access_event(self,
                        model_id: str,
                        access_type: str,
                        user_id: Optional[str] = None,
                        metadata: Dict[str, Any] = None) -> str:
        """Log a model access event."""
        event_id = str(uuid.uuid4())
        
        event = AuditEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type="access",
            user_id=user_id,
            model_id=model_id,
            input_data_hash="",
            output_data={"access_type": access_type},
            metadata=metadata or {},
            compliance_tags=["access"],
            data_lineage={}
        )
        
        self._store_audit_event(event)
        self._check_compliance(event)
        
        return event_id
    
    def _hash_data(self, data: str) -> str:
        """Create hash of data for privacy and integrity."""
        return hashlib.sha256(data.encode()).hexdigest()
    
    def _create_data_lineage(self, data: Any, source_type: str) -> Dict[str, Any]:
        """Create data lineage information."""
        return {
            "source_type": source_type,
            "timestamp": datetime.now().isoformat(),
            "transformations": [],
            "data_hash": self._hash_data(str(data))
        }
    
    def _store_audit_event(self, event: AuditEvent):
        """Store audit event in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO audit_events 
            (event_id, timestamp, event_type, user_id, model_id, 
             input_data_hash, output_data, metadata, compliance_tags, data_lineage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            event.event_id,
            event.timestamp.isoformat(),
            event.event_type,
            event.user_id,
            event.model_id,
            event.input_data_hash,
            json.dumps(event.output_data),
            json.dumps(event.metadata),
            json.dumps(event.compliance_tags),
            json.dumps(event.data_lineage)
        ))
        
        conn.commit()
        conn.close()
        
        # Update statistics
        self.audit_stats['total_events'] += 1
        self.audit_stats['events_by_type'][event.event_type] += 1
    
    def _check_compliance(self, event: AuditEvent):
        """Check event against compliance rules."""
        for rule_id, rule in self.compliance_rules.items():
            if not rule.enabled:
                continue
            
            violation = self._evaluate_rule(rule, event)
            if violation:
                self._log_compliance_violation(rule_id, event.event_id, violation)
    
    def _evaluate_rule(self, rule: ComplianceRule, event: AuditEvent) -> Optional[str]:
        """Evaluate a compliance rule against an event."""
        if rule.rule_type == "retention":
            # Check if old data should be deleted
            age_days = (datetime.now() - event.timestamp).days
            if age_days > rule.parameters.get("retention_days", 365):
                return f"Event exceeds retention period of {rule.parameters['retention_days']} days"
        
        elif rule.rule_type == "bias" and event.event_type == "prediction":
            # Simplified bias check
            if self._check_bias_in_prediction(event):
                return "Potential bias detected in prediction"
        
        elif rule.rule_type == "access":
            # Log all access events (no violation, just logging)
            if event.event_type == "access":
                return None  # No violation, just ensure logging
        
        return None
    
    def _check_bias_in_prediction(self, event: AuditEvent) -> bool:
        """Check for potential bias in prediction (simplified implementation)."""
        # This would be much more sophisticated in a real implementation
        # involving protected attributes, demographic parity, etc.
        
        confidence = event.output_data.get("confidence", 1.0)
        
        # Flag low-confidence predictions for review
        if confidence < 0.5:
            return True
        
        return False
    
    def _log_compliance_violation(self, rule_id: str, event_id: str, description: str):
        """Log a compliance violation."""
        violation_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        rule = self.compliance_rules[rule_id]
        
        cursor.execute('''
            INSERT INTO compliance_violations 
            (violation_id, timestamp, rule_id, event_id, severity, description)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            violation_id,
            datetime.now().isoformat(),
            rule_id,
            event_id,
            rule.severity,
            description
        ))
        
        conn.commit()
        conn.close()
        
        self.audit_stats['compliance_violations'] += 1
        self.logger.warning(f"Compliance violation: {description}")
    
    def create_gdpr_request(self, 
                           request_type: str,
                           user_id: Optional[str] = None,
                           email: Optional[str] = None) -> str:
        """
        Create a GDPR request (right to access, right to be forgotten, etc.).
        
        Args:
            request_type: Type of request (access, deletion, portability, etc.)
            user_id: User identifier
            email: User email
            
        Returns:
            Request ID
        """
        if not self.enable_gdpr:
            raise ValueError("GDPR features not enabled")
        
        request_id = str(uuid.uuid4())
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO gdpr_requests 
            (request_id, request_type, user_id, email, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            request_id,
            request_type,
            user_id,
            email,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        self.audit_stats['gdpr_requests'] += 1
        self.logger.info(f"GDPR request created: {request_type} for {user_id or email}")
        
        return request_id
    
    def process_gdpr_request(self, request_id: str) -> Dict[str, Any]:
        """Process a GDPR request."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get request details
        cursor.execute('''
            SELECT request_type, user_id, email FROM gdpr_requests 
            WHERE request_id = ?
        ''', (request_id,))
        
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"GDPR request {request_id} not found")
        
        request_type, user_id, email = result
        
        if request_type == "access":
            # Right to access - return user's data
            user_data = self._get_user_data(user_id, email)
            response = {"data": user_data}
            
        elif request_type == "deletion":
            # Right to be forgotten - delete user's data
            deleted_count = self._delete_user_data(user_id, email)
            response = {"deleted_records": deleted_count}
            
        elif request_type == "portability":
            # Right to data portability - export user's data
            user_data = self._get_user_data(user_id, email)
            response = {"exportable_data": user_data}
            
        else:
            response = {"error": f"Unsupported request type: {request_type}"}
        
        # Mark request as processed
        cursor.execute('''
            UPDATE gdpr_requests 
            SET status = 'completed', processed_timestamp = ?
            WHERE request_id = ?
        ''', (datetime.now().isoformat(), request_id))
        
        conn.commit()
        conn.close()
        
        return response
    
    def _get_user_data(self, user_id: Optional[str], email: Optional[str]) -> Dict[str, Any]:
        """Get all data for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get audit events for user
        cursor.execute('''
            SELECT * FROM audit_events 
            WHERE user_id = ? OR user_id = ?
        ''', (user_id, email))
        
        events = [dict(zip([col[0] for col in cursor.description], row)) 
                 for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "user_id": user_id,
            "email": email,
            "audit_events": events,
            "total_events": len(events)
        }
    
    def _delete_user_data(self, user_id: Optional[str], email: Optional[str]) -> int:
        """Delete all data for a user."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete audit events
        cursor.execute('''
            DELETE FROM audit_events 
            WHERE user_id = ? OR user_id = ?
        ''', (user_id, email))
        
        deleted_count = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Deleted {deleted_count} records for user {user_id or email}")
        return deleted_count
    
    def cleanup_expired_data(self) -> int:
        """Clean up data that exceeds retention period."""
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete old audit events
        cursor.execute('''
            DELETE FROM audit_events 
            WHERE datetime(timestamp) < datetime(?)
        ''', (cutoff_date.isoformat(),))
        
        deleted_count = cursor.rowcount
        
        # Delete old compliance violations
        cursor.execute('''
            DELETE FROM compliance_violations 
            WHERE datetime(timestamp) < datetime(?)
        ''', (cutoff_date.isoformat(),))
        
        deleted_count += cursor.rowcount
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleaned up {deleted_count} expired records")
        return deleted_count
    
    def get_audit_report(self, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        event_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        if start_date is None:
            start_date = datetime.now() - timedelta(days=30)
        if end_date is None:
            end_date = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        
        # Base query
        where_clauses = ["datetime(timestamp) BETWEEN datetime(?) AND datetime(?)"]
        params = [start_date.isoformat(), end_date.isoformat()]
        
        if event_types:
            where_clauses.append(f"event_type IN ({','.join(['?' for _ in event_types])})")
            params.extend(event_types)
        
        where_clause = " AND ".join(where_clauses)
        
        # Get event statistics
        query = f'''
            SELECT event_type, COUNT(*) as count
            FROM audit_events 
            WHERE {where_clause}
            GROUP BY event_type
        '''
        event_stats = dict(pd.read_sql_query(query, conn, params=params).values)
        
        # Get user activity
        query = f'''
            SELECT user_id, COUNT(*) as count
            FROM audit_events 
            WHERE {where_clause} AND user_id IS NOT NULL
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        '''
        user_activity = dict(pd.read_sql_query(query, conn, params=params).values)
        
        # Get model usage
        query = f'''
            SELECT model_id, COUNT(*) as count
            FROM audit_events 
            WHERE {where_clause}
            GROUP BY model_id
            ORDER BY count DESC
        '''
        model_usage = dict(pd.read_sql_query(query, conn, params=params).values)
        
        # Get compliance violations
        query = f'''
            SELECT rule_id, severity, COUNT(*) as count
            FROM compliance_violations 
            WHERE datetime(timestamp) BETWEEN datetime(?) AND datetime(?)
            GROUP BY rule_id, severity
        '''
        violations_df = pd.read_sql_query(query, conn, params=[start_date.isoformat(), end_date.isoformat()])
        violations = violations_df.to_dict('records') if not violations_df.empty else []
        
        conn.close()
        
        return {
            "report_period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "summary": {
                "total_events": sum(event_stats.values()),
                "unique_users": len(user_activity),
                "models_accessed": len(model_usage),
                "compliance_violations": len(violations)
            },
            "event_statistics": event_stats,
            "top_users": user_activity,
            "model_usage": model_usage,
            "compliance_violations": violations,
            "gdpr_requests": self.audit_stats['gdpr_requests']
        }
    
    def get_compliance_status(self) -> Dict[str, Any]:
        """Get current compliance status."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent violations
        cursor.execute('''
            SELECT severity, COUNT(*) as count
            FROM compliance_violations 
            WHERE datetime(timestamp) > datetime(?)
            GROUP BY severity
        ''', ((datetime.now() - timedelta(days=7)).isoformat(),))
        
        recent_violations = dict(cursor.fetchall())
        
        # Get unresolved violations
        cursor.execute('''
            SELECT COUNT(*) FROM compliance_violations 
            WHERE resolved = FALSE
        ''')
        
        unresolved_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "compliance_rules": {
                rule_id: {
                    "name": rule.name,
                    "enabled": rule.enabled,
                    "severity": rule.severity
                }
                for rule_id, rule in self.compliance_rules.items()
            },
            "recent_violations": recent_violations,
            "unresolved_violations": unresolved_count,
            "data_retention_days": self.retention_days,
            "gdpr_enabled": self.enable_gdpr,
            "bias_monitoring_enabled": self.enable_bias_monitoring
        }
    
    def export_audit_data(self, filepath: str, days: int = 30):
        """Export audit data to CSV."""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        conn = sqlite3.connect(self.db_path)
        query = '''
            SELECT * FROM audit_events 
            WHERE datetime(timestamp) > datetime(?)
            ORDER BY timestamp DESC
        '''
        df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
        
        df.to_csv(filepath, index=False)
        conn.close()
        
        self.logger.info(f"Exported {len(df)} audit records to {filepath}")
