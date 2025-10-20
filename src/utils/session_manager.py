#!/usr/bin/env python3
"""
Session Manager for Continuous Interactive Mode

Manages multiple research sessions in continuous interactive mode, including:
- Session lifecycle management
- Real-time status monitoring
- Result parsing and storage
- Data sharing between sessions
"""

import json
import os
import subprocess
import threading
import time
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import re


class SessionManager:
    """Research session manager for multi-session support"""
    
    def __init__(self, base_dir: Path = None):
        self.base_dir = base_dir or Path.cwd()
        self.sessions = {}  # Active sessions
        self.completed_sessions = {}  # Completed sessions
        self.session_counter = 0
        
        # Create session directory
        self.sessions_dir = self.base_dir / 'sessions'
        self.sessions_dir.mkdir(exist_ok=True)
        
        # Output directory
        self.outputs_dir = self.base_dir / 'outputs'
        self.outputs_dir.mkdir(exist_ok=True)
    
    def create_session(self, topic: str, num_ideas: int = 3, debug_mode: bool = False) -> str:
        """Create a new research session"""
        self.session_counter += 1
        session_id = f"session_{self.session_counter}_{datetime.now().strftime('%H%M%S')}"
        
        # Session directory
        session_dir = self.sessions_dir / session_id
        session_dir.mkdir(exist_ok=True)
        
        # Build command
        cmd = [
            'python', '-m', 'src',
            '--topic', topic,
            '--num_ideas', str(num_ideas)
        ]
        
        if debug_mode:
            cmd.append('--debug')
        
        # Session information
        session_info = {
            'id': session_id,
            'topic': topic,
            'num_ideas': num_ideas,
            'debug_mode': debug_mode,
            'start_time': datetime.now(),
            'status': 'initializing',
            'phase': 'Initializing',
            'progress': 0,
            'total_steps': num_ideas * 4 + 5,  # Estimated step count
            'current_step': 'Starting research process...',
            'logs': [],
            'literature_papers': [],
            'generated_ideas': [],
            'evaluation_results': {},
            'final_output': None,
            'error': None,
            'session_dir': session_dir,
            'process': None,
            'output_queue': queue.Queue(),
            'monitor_thread': None
        }
        
        self.sessions[session_id] = session_info
        return session_id
    
    def start_session(self, session_id: str) -> bool:
        """Start session process"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            # Start subprocess with proper environment for real-time output
            env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
            process = subprocess.Popen(
                [
                    'python', '-m', 'src',
                    '--topic', session['topic'],
                    '--num_ideas', str(session['num_ideas'])
                ] + (['--debug'] if session['debug_mode'] else []),
                cwd=self.base_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout for combined output
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            session['process'] = process
            session['status'] = 'running'
            
            # Start monitoring thread
            monitor_thread = threading.Thread(
                target=self._monitor_process_output,
                args=(session_id,),
                daemon=True
            )
            monitor_thread.start()
            session['monitor_thread'] = monitor_thread
            
            return True
            
        except Exception as e:
            session['status'] = 'error'
            session['error'] = str(e)
            return False
    
    def _monitor_process_output(self, session_id: str):
        """Monitor process output with real-time streaming"""
        session = self.sessions[session_id]
        process = session['process']
        
        try:
            # Add initial log
            self._process_output_line(session_id, f"🚀 Research process started (PID: {process.pid})")
            
            while process.poll() is None:
                try:
                    # Try to read a line with a timeout
                    line = process.stdout.readline()
                    if line:
                        stripped_line = line.strip()
                        if stripped_line:  # Only process non-empty lines
                            self._process_output_line(session_id, stripped_line)
                    else:
                        # No output available, small sleep to prevent busy waiting
                        time.sleep(0.1)
                        
                except (OSError, ValueError) as e:
                    # Process might have ended or be unavailable
                    self._process_output_line(session_id, f"⚠️ Output reading interrupted: {e}")
                    break
            
            # Process ended, read remaining output
            try:
                remaining_output = process.stdout.read()
                if remaining_output:
                    for line in remaining_output.splitlines():
                        stripped_line = line.strip()
                        if stripped_line:
                            self._process_output_line(session_id, stripped_line)
            except Exception as e:
                self._process_output_line(session_id, f"⚠️ Failed to read remaining output: {e}")
            
            # Update session status
            return_code = process.returncode
            if return_code == 0:
                session['status'] = 'completed'
                self._process_output_line(session_id, "✅ Research completed successfully!")
                self._finalize_session(session_id)
            else:
                session['status'] = 'error'
                session['error'] = f"Process exited with code {return_code}"
                self._process_output_line(session_id, f"❌ Research failed with exit code {return_code}")
            
        except Exception as e:
            session['status'] = 'error'
            session['error'] = f"Monitor error: {e}"
            self._process_output_line(session_id, f"💥 Process monitoring error: {e}")
    
    def _process_output_line(self, session_id: str, line: str):
        """Process single line of output"""
        session = self.sessions[session_id]
        
        # Add to logs
        log_entry = {
            'timestamp': datetime.now(),
            'content': line
        }
        session['logs'].append(log_entry)
        
        # Parse special content
        self._parse_output_content(session_id, line)
        
        # Keep log size reasonable
        if len(session['logs']) > 1000:
            session['logs'] = session['logs'][-500:]
    
    def _parse_output_content(self, session_id: str, line: str):
        """Parse output content and extract structured information"""
        session = self.sessions[session_id]

        # Detect when pipeline starts
        if 'Starting Orchestrated Research Pipeline' in line:
            session['phase'] = 'Pipeline Starting'
            session['current_step'] = 'Initializing research pipeline'
            session['status'] = 'running'

        # Parse actual phase format: "Phase 0:", "Phase 1:", etc.
        phase_match = re.search(r'Phase\s+(\d+):\s*(.+)', line)
        if phase_match:
            phase_num = int(phase_match.group(1))
            phase_desc = phase_match.group(2).strip()

            # Map phase numbers to display names
            phase_map = {
                0: 'System Validation',
                1: 'Literature Search',
                2: 'Idea Generation',
                3: 'Internal Selection',
                4: 'Distinctness Analysis',
                5: 'Multi-Agent Review',
                6: 'Final Selection',
                7: 'Portfolio Analysis'
            }

            session['phase'] = phase_map.get(phase_num, f"Phase {phase_num}")
            session['current_step'] = phase_desc
            session['status'] = 'running'

        # Also detect KG building phase within Idea Generation
        if 'Building literature-informed knowledge graph' in line or 'knowledge graph construction' in line:
            session['current_step'] = 'Building Knowledge Graph'
        
        # Parse literature search with multiple patterns
        literature_patterns = [
            r'Found (\d+) papers?',
            r'Retrieved (\d+) papers?',
            r'(\d+) papers? found',
            r'Literature search returned (\d+)',
            r'Papers retrieved: (\d+)'
        ]
        
        for pattern in literature_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                count = int(match.group(1))
                session['literature_count'] = count
                session['current_step'] = f"Found {count} relevant papers"
                break
        
        # Parse idea generation with better patterns
        idea_patterns = [
            r'Generated idea (\d+)',
            r'Idea (\d+) generated',
            r'Creating idea (\d+)',
            r'\[Idea (\d+)\]'
        ]
        
        for pattern in idea_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                idea_num = int(match.group(1))
                session['current_step'] = f"Generated idea {idea_num}"
                session['progress'] = min(40 + (idea_num * 10), 80)  # Progress based on idea count
                break
        
        # Parse evaluation and review phases
        evaluation_patterns = [
            r'Evaluation completed',
            r'Review completed',
            r'Multi-agent review finished',
            r'Portfolio analysis complete'
        ]
        
        for pattern in evaluation_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                session['current_step'] = "Evaluation completed"
                session['progress'] = 90
                break
        
        # Parse completion
        if any(phrase in line.lower() for phrase in ['research completed', 'process finished', 'analysis complete']):
            session['current_step'] = "Research completed successfully"
            session['progress'] = 100
        
        # Parse errors with better detection
        error_patterns = [
            r'Error:(.+)',
            r'Failed to (.+)',
            r'Exception:(.+)',
            r'Traceback',
            r'ERROR'
        ]
        
        for pattern in error_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match and session['error'] is None:
                session['error'] = line.strip()
                session['current_step'] = "Error encountered"
                break
    
    def _finalize_session(self, session_id: str):
        """Finalize session processing"""
        session = self.sessions[session_id]
        
        try:
            # Find output files
            output_files = list(self.outputs_dir.glob(f"*{session['topic'].replace(' ', '_')}*.json"))
            
            if output_files:
                # Find the latest output file
                latest_file = max(output_files, key=lambda x: x.stat().st_mtime)
                
                # Parse results
                with open(latest_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                session['final_output'] = results
                session['generated_ideas'] = results.get('generated_ideas', [])
                session['evaluation_results'] = results.get('evaluation_results', {})
                
                # Extract literature information
                if 'literature_search' in results:
                    lit_search = results['literature_search']
                    session['literature_papers'] = lit_search.get('papers', [])
            
            session['end_time'] = datetime.now()
            session['duration'] = (session['end_time'] - session['start_time']).total_seconds()
            
            # Move to completed sessions
            self.completed_sessions[session_id] = session
            
        except Exception as e:
            session['error'] = f"Finalization error: {e}"
    
    def get_session_status(self, session_id: str) -> Optional[Dict]:
        """Get session status"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            return {
                'id': session_id,
                'topic': session['topic'],
                'status': session['status'],
                'phase': session['phase'],
                'progress': session['progress'],
                'current_step': session['current_step'],
                'start_time': session['start_time'].isoformat(),
                'error': session['error']
            }
        elif session_id in self.completed_sessions:
            session = self.completed_sessions[session_id]
            return {
                'id': session_id,
                'topic': session['topic'],
                'status': 'completed',
                'phase': 'finished',
                'progress': 100,
                'current_step': 'Session completed',
                'start_time': session['start_time'].isoformat(),
                'end_time': session.get('end_time', datetime.now()).isoformat(),
                'duration': session.get('duration', 0),
                'error': session['error']
            }
        
        return None
    
    def get_session_logs(self, session_id: str, limit: int = 50) -> List[str]:
        """Get session logs with real-time updates"""
        session = self.sessions.get(session_id) or self.completed_sessions.get(session_id)
        if not session:
            return [f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Session {session_id} not found"]
        
        logs = session['logs'][-limit:] if session['logs'] else []
        
        # Always include basic session info
        status = session.get('status', 'unknown')
        topic = session.get('topic', 'Unknown')
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        if not logs:
            # Return status info if no logs yet
            return [
                f"[{timestamp}] 🔍 Session {session_id} ({status})",
                f"[{timestamp}] 📝 Topic: {topic}",
                f"[{timestamp}] ⏳ Total logs in session: 0",
                f"[{timestamp}] 🔄 Waiting for process output..."
            ]
        
        formatted_logs = []
        
        # Add session header
        formatted_logs.append(f"[{timestamp}] 📊 Session {session_id} - {len(logs)} log entries")
        
        # Add actual logs
        for log in logs:
            log_timestamp = log['timestamp'].strftime('%H:%M:%S')
            content = log['content']
            formatted_logs.append(f"[{log_timestamp}] {content}")
        
        # Add current status footer
        formatted_logs.append(f"[{timestamp}] 📈 Current status: {status}")
        
        return formatted_logs
    
    def get_session_results(self, session_id: str) -> Dict[str, Any]:
        """Get session result data"""
        session = self.sessions.get(session_id) or self.completed_sessions.get(session_id)
        if not session:
            return {'literature': pd.DataFrame(), 'ideas': pd.DataFrame(), 
                   'evaluation': {}, 'final_output': {}}
        
        # Literature DataFrame
        literature_df = pd.DataFrame()
        if session['literature_papers']:
            lit_data = []
            for paper in session['literature_papers']:
                lit_data.append([
                    paper.get('title', 'Unknown'),
                    ', '.join(paper.get('authors', [])),
                    paper.get('year', 'N/A'),
                    paper.get('citationCount', 0)
                ])
            literature_df = pd.DataFrame(lit_data, columns=['Title', 'Authors', 'Year', 'Citations'])
        
        # Ideas DataFrame
        ideas_df = pd.DataFrame()
        if session['generated_ideas']:
            ideas_data = []
            for i, idea in enumerate(session['generated_ideas']):
                ideas_data.append([
                    i,
                    idea.get('title', 'Untitled'),
                    'completed' if session['status'] == 'completed' else 'generating',
                    idea.get('novelty_score', 'N/A'),
                    idea.get('feasibility_score', 'N/A')
                ])
            ideas_df = pd.DataFrame(ideas_data, columns=['ID', 'Title', 'Status', 'Novelty', 'Feasibility'])
        
        return {
            'literature': literature_df,
            'ideas': ideas_df,
            'evaluation': session['evaluation_results'],
            'final_output': session['final_output'] or {}
        }
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate session"""
        if session_id not in self.sessions:
            return False
        
        session = self.sessions[session_id]
        
        try:
            if session['process'] and session['process'].poll() is None:
                session['process'].terminate()
                
                # Wait for process to end
                try:
                    session['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    session['process'].kill()
            
            session['status'] = 'terminated'
            session['end_time'] = datetime.now()
            
            # Move to completed sessions
            self.completed_sessions[session_id] = session
            del self.sessions[session_id]
            
            return True
            
        except Exception as e:
            session['error'] = f"Termination error: {e}"
            return False
    
    def list_active_sessions(self) -> List[str]:
        """List active sessions"""
        return list(self.sessions.keys())
    
    def list_completed_sessions(self) -> List[str]:
        """List completed sessions"""
        return list(self.completed_sessions.keys())
    
    def cleanup_old_sessions(self, max_age_hours: int = 24):
        """Cleanup old sessions"""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        # Clean up old completed sessions
        to_remove = []
        for session_id, session in self.completed_sessions.items():
            session_time = session['start_time'].timestamp()
            if session_time < cutoff_time:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            del self.completed_sessions[session_id]
        
        return len(to_remove)
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session summary"""
        active_count = len(self.sessions)
        completed_count = len(self.completed_sessions)
        
        # Count various statuses
        status_counts = {}
        for session in self.sessions.values():
            status = session['status']
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            'total_sessions': self.session_counter,
            'active_sessions': active_count,
            'completed_sessions': completed_count,
            'status_breakdown': status_counts,
            'last_session_id': max(self.sessions.keys()) if self.sessions else None
        }


# Global session manager instance
_session_manager = None

def get_session_manager() -> SessionManager:
    """Get global session manager instance"""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager