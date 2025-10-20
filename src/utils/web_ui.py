#!/usr/bin/env python3
"""
Web UI for Autonomous Research Agent Process Visualization

Provides real-time visualization of the research agent's execution phases,
idea generation progress, and final results through a clean Gradio interface.
"""

import json
import asyncio
import threading
import time
import random
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import gradio as gr
import uuid
import pandas as pd
from .session_manager import SessionManager




# Global UI instance for easy access
_ui_instance = None
def get_ui_instance() -> 'ProcessVisualizerUI':
    """Get or create the global UI instance"""
    global _ui_instance
    if _ui_instance is None:
        _ui_instance = ProcessVisualizerUI()
    return _ui_instance

def start_web_ui(host='localhost', port=7860, share=False) -> 'ProcessVisualizerUI':
    """Start the Gradio web UI server (host configurable, default localhost)"""
    global _ui_instance
    _ui_instance = ProcessVisualizerUI(port=port, share=share, host=host)
    _ui_instance.launch()
    return _ui_instance



class ProcessVisualizerUI:
    """Complete process visualization UI with multi-session management"""
    
    def __init__(self, config: Dict = None, port: int = 7860, share: bool = False, host: str = 'localhost'):
        """Initialize the process visualization UI"""
        self.config = config or {}
        self.port = port
        self.share = share
        self.host = host or 'localhost'
        self.interface = None
        self.current_session_id = None
        self.session_manager = SessionManager()
        self.active_sessions = {}  # Legacy support
        self.base_dir = Path.cwd()
    
    def create_interface(self):
        """Create the complete Gradio interface"""
        # Custom CSS for better large screen support and Times New Roman font
        custom_css = """
        /* Global font setting */
        * {
            font-family: 'Times New Roman', Times, serif !important;
        }

        /* Remove max-width constraints for large screens */
        .gradio-container {
            max-width: 100% !important;
            padding: 1.5rem 3rem !important;
        }
        .main {
            max-width: 100% !important;
        }
        .contain {
            max-width: 100% !important;
        }

        /* Base font size increase */
        body {
            font-size: 16px !important;
        }

        /* Responsive font sizing for large screens */
        @media (min-width: 1920px) {
            .gradio-container {
                font-size: 18px !important;
                padding: 2rem 4rem !important;
            }
            h1 {
                font-size: 3rem !important;
            }
            h2 {
                font-size: 1.75rem !important;
                margin-bottom: 1rem !important;
            }
            label {
                font-size: 1.1rem !important;
            }
        }

        @media (min-width: 2560px) {
            .gradio-container {
                font-size: 20px !important;
                padding: 3rem 6rem !important;
            }
            h1 {
                font-size: 3.5rem !important;
            }
            h2 {
                font-size: 2rem !important;
            }
            label {
                font-size: 1.2rem !important;
            }
        }

        /* Larger input fields on big screens */
        @media (min-width: 2560px) {
            .gr-textbox input,
            .gr-textbox textarea {
                font-size: 1.1rem !important;
                padding: 0.75rem !important;
            }
            .gr-button {
                font-size: 1.1rem !important;
            }
        }

        /* Make output and logs containers fill available space */
        .output-container {
            min-height: 450px !important;
        }

        .logs-container textarea {
            height: 350px !important;
            max-height: 350px !important;
            font-family: 'Courier New', Courier, monospace !important;
            resize: none !important;
        }

        /* Better spacing for groups */
        .gr-group {
            margin-bottom: 1.25rem !important;
            padding: 1rem !important;
        }

        /* Ensure JSON viewer is scrollable and fills space */
        .gr-json {
            max-height: 650px !important;
            overflow-y: auto !important;
            font-size: 0.95em !important;
        }

        /* 2K+ screens - only increase font sizes, not heights */
        @media (min-width: 2560px) {
            .logs-container textarea {
                font-size: 1.05rem !important;
            }

            .gr-json {
                font-size: 1em !important;
            }
        }

        /* Better button sizing on large screens */
        @media (min-width: 1920px) {
            .gr-button {
                padding: 0.75rem 1.5rem !important;
                font-size: 1rem !important;
            }
        }

        @media (min-width: 2560px) {
            .gr-button {
                padding: 1rem 2rem !important;
                font-size: 1.1rem !important;
                min-height: 48px !important;
            }
        }

        /* Ensure columns have equal height */
        .gr-row > .gr-column {
            display: flex !important;
            flex-direction: column !important;
        }

        /* Make textboxes more readable on large screens */
        @media (min-width: 1920px) {
            .gr-textbox input,
            .gr-textbox textarea {
                font-size: 1rem !important;
            }
        }

        @media (min-width: 2560px) {
            .gr-textbox input,
            .gr-textbox textarea {
                font-size: 1.1rem !important;
            }
        }

        /* Larger dropdowns and sliders on 2K+ */
        @media (min-width: 2560px) {
            .gr-dropdown,
            .gr-slider {
                font-size: 1.1rem !important;
            }

            .gr-dropdown .wrap {
                min-height: 48px !important;
            }
        }

        /* Theme toggle button styling */
        #theme-toggle-btn {
            min-width: 140px !important;
        }

        @media (min-width: 2560px) {
            #theme-toggle-btn {
                min-width: 180px !important;
                font-size: 1.1rem !important;
            }
        }

        /* Dark theme styles using data-theme attribute */
        body[data-theme="dark"],
        body[data-theme="dark"] .gradio-container,
        body.dark,
        body.dark .gradio-container {
            background-color: #0b0f19 !important;
            color: #e5e7eb !important;
        }

        body[data-theme="dark"] .gr-group,
        body.dark .gr-group {
            background-color: #1f2937 !important;
            border-color: #374151 !important;
        }

        body[data-theme="dark"] .gr-input,
        body[data-theme="dark"] .gr-textbox,
        body[data-theme="dark"] textarea,
        body[data-theme="dark"] input,
        body.dark .gr-input,
        body.dark .gr-textbox,
        body.dark textarea,
        body.dark input {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            border-color: #4b5563 !important;
        }

        body[data-theme="dark"] .gr-button,
        body.dark .gr-button {
            background-color: #374151 !important;
            color: #e5e7eb !important;
            border-color: #4b5563 !important;
        }

        body[data-theme="dark"] .gr-button:hover,
        body.dark .gr-button:hover {
            background-color: #4b5563 !important;
        }

        body[data-theme="dark"] h1,
        body[data-theme="dark"] h2,
        body[data-theme="dark"] h3,
        body.dark h1,
        body.dark h2,
        body.dark h3 {
            color: #f3f4f6 !important;
        }

        body[data-theme="dark"] label,
        body.dark label {
            color: #d1d5db !important;
        }

        body[data-theme="dark"] .gr-json,
        body.dark .gr-json {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
        }

        body[data-theme="dark"] .gr-dropdown,
        body.dark .gr-dropdown {
            background-color: #1f2937 !important;
            color: #e5e7eb !important;
            border-color: #4b5563 !important;
        }
        """

        with gr.Blocks(title="AutoResearcher", theme=gr.themes.Soft(), css=custom_css) as interface:
            # Theme toggle button at the top right
            with gr.Row():
                with gr.Column(scale=10):
                    gr.Markdown("# 🔬 AutoResearcher")
                with gr.Column(scale=1, min_width=150):
                    theme_toggle = gr.Button("🌙 Dark Mode", elem_id="theme-toggle-btn", size="sm")

            with gr.Row(equal_height=True):
                with gr.Column(scale=1, min_width=400):
                    # Session management
                    with gr.Group():
                        gr.Markdown("## 📋 Session Management")
                        session_list = gr.Dropdown(
                            label="Research Sessions",
                            choices=[],
                            info="Select a research session to view"
                        )

                        with gr.Row():
                            refresh_sessions_btn = gr.Button("🔄 Refresh", size="sm")
                            terminate_session_btn = gr.Button("⛔ Terminate Session", size="sm", variant="stop")

                    # New session creation
                    with gr.Group():
                        gr.Markdown("## ➕ New Session")
                        topic_input = gr.Textbox(
                            label="Research Topic",
                            placeholder="Enter research topic...",
                            lines=3
                        )
                        num_ideas = gr.Slider(
                            minimum=1, maximum=10, value=3, step=1,
                            label="Number of Ideas"
                        )

                        start_research_btn = gr.Button("🚀 Start Research", variant="primary", size="lg")

                    # Session status
                    with gr.Group():
                        gr.Markdown("## 📊 Current Session Status")
                        session_status = gr.Textbox(
                            label="Session Status",
                            value="No active session",
                            interactive=False
                        )
                        current_phase = gr.Textbox(
                            label="Current Phase",
                            value="Idle",
                            interactive=False
                        )
                        progress_info = gr.Textbox(
                            label="Progress Information",
                            value="Waiting to start...",
                            interactive=False,
                            lines=2
                        )
                
                with gr.Column(scale=2):
                    # Final output display
                    with gr.Group():
                        gr.Markdown("## 📋 Final Output")
                        final_output_display = gr.JSON(
                            label="Complete Research Results",
                            show_label=False,
                            container=True,
                            elem_classes="output-container"
                        )

                    # Real-time logs
                    with gr.Group():
                        gr.Markdown("## 📝 Real-time Activity Logs")
                        logs_display = gr.Textbox(
                            label="Activity Logs",
                            lines=15,
                            interactive=False,
                            show_copy_button=True,
                            elem_classes="logs-container"
                        )
            
            
            
            # Event handlers

            # Theme toggle handler using Gradio's native approach
            theme_toggle.click(
                fn=None,
                js="""() => {
                    console.log('Theme toggle clicked!');

                    // Find the Gradio app root
                    const gradioContainer = document.querySelector('.gradio-container');
                    const body = document.body;
                    const btn = document.getElementById('theme-toggle-btn');

                    // Check current theme
                    const isDark = body.getAttribute('data-theme') === 'dark';
                    console.log('Current isDark:', isDark);

                    if (isDark) {
                        // Switch to light
                        body.setAttribute('data-theme', 'light');
                        body.classList.remove('dark');
                        if (gradioContainer) gradioContainer.classList.remove('dark');
                        if (btn) btn.textContent = '🌙 Dark Mode';
                        localStorage.setItem('theme-mode', 'light');
                        console.log('Switched to light mode');
                    } else {
                        // Switch to dark
                        body.setAttribute('data-theme', 'dark');
                        body.classList.add('dark');
                        if (gradioContainer) gradioContainer.classList.add('dark');
                        if (btn) btn.textContent = '☀️ Light Mode';
                        localStorage.setItem('theme-mode', 'dark');
                        console.log('Switched to dark mode');
                    }
                }"""
            )

            start_research_btn.click(
                fn=self._start_research_session,
                inputs=[topic_input, num_ideas],
                outputs=[session_status, current_phase, progress_info, final_output_display, logs_display, session_list]
            )
            
            session_list.change(
                fn=self._switch_session,
                inputs=[session_list],
                outputs=[session_status, current_phase, progress_info, final_output_display, logs_display]
            )
            
            terminate_session_btn.click(
                fn=self._terminate_session,
                inputs=[session_list],
                outputs=[session_status, current_phase, progress_info, final_output_display, logs_display, session_list]
            )
            
            # Set up automatic refresh timer - updates every 1 second
            timer = gr.Timer(value=1.0, active=True)
            timer.tick(
                fn=self._auto_refresh_display,
                inputs=[],  # No inputs needed - use internal session tracking
                outputs=[session_status, current_phase, progress_info, final_output_display, logs_display]
            )

            # Also refresh session list periodically but preserve selection
            timer.tick(
                fn=self._auto_refresh_sessions,
                inputs=[],
                outputs=[session_list]
            )
            
            
            # No need for manual refresh thread anymore - using Gradio's timer
        
        return interface
    
    def _add_session_log(self, session_id: str, message: str):
        """Add a log message to a specific session and parse structured data"""
        # Add to both active_sessions (legacy) and session manager
        if session_id in self.active_sessions:
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_entry = f"[{timestamp}] {message}"
            self.active_sessions[session_id]['logs'].append(log_entry)
            
            # Keep only last 100 logs per session
            if len(self.active_sessions[session_id]['logs']) > 100:
                self.active_sessions[session_id]['logs'] = self.active_sessions[session_id]['logs'][-100:]
            
            # Parse structured data from the message
            self._parse_message_for_data(session_id, message)
        
        # Also update the session manager if it has this session
        if (session_id in self.session_manager.sessions or 
            session_id in self.session_manager.completed_sessions):
            # Session manager handles its own logging, so we sync the parsed data
            self._sync_session_data(session_id)
    
    def _parse_message_for_data(self, session_id: str, message: str):
        """Parse log message for structured data (literature, ideas, phases)"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        
        # Parse phase information
        if 'Phase' in message:
            if 'Literature' in message or 'Phase 0' in message:
                session['current_phase'] = 'Literature Search'
                session['progress'] = 20
            elif 'Idea Generation' in message or 'Phase 1' in message:
                session['current_phase'] = 'Idea Generation'
                session['progress'] = 40
            elif 'Distinctness' in message or 'Phase 1.5' in message:
                session['current_phase'] = 'Distinctness Analysis'
                session['progress'] = 60
            elif 'Review' in message or 'Phase 2' in message:
                session['current_phase'] = 'Multi-Agent Review'
                session['progress'] = 80
            elif 'Portfolio' in message or 'Phase 3' in message:
                session['current_phase'] = 'Portfolio Analysis'
                session['progress'] = 100
        
        # Try to load actual output files to get real data
        self._load_real_output_data(session_id)
        
    def _load_real_output_data(self, session_id: str):
        """Load actual output data from files with improved parsing"""
        if session_id not in self.active_sessions:
            return
            
        session = self.active_sessions[session_id]
        topic = session.get('topic', '')
        
        # Try to find output files for this topic
        try:
            output_dir = Path('outputs')
            if not output_dir.exists():
                return
                
            # Look for files matching the topic pattern (be more flexible with timing)
            topic_pattern = topic.replace(' ', '_').lower()
            matching_files = []
            
            for json_file in output_dir.glob('*.json'):
                # Check if file contains our topic or was created recently
                if (topic_pattern in json_file.name.lower() or 
                    (time.time() - json_file.stat().st_mtime < 1800)):  # 30 minutes
                    matching_files.append(json_file)
            
            if matching_files:
                # Get the most recent file
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract literature papers with better structure
                if 'literature_search_results' in data:
                    lit_data = data['literature_search_results']
                    papers = lit_data.get('papers', [])
                    if papers:
                        session['literature_papers'] = papers[:15]  # Show more papers
                        session['literature_count'] = len(papers)
                
                # Handle different literature search result formats
                elif 'semantic_scholar_results' in data:
                    papers = data['semantic_scholar_results'].get('papers', [])
                    if papers:
                        session['literature_papers'] = papers[:15]
                        session['literature_count'] = len(papers)
                
                # Extract generated ideas with better status tracking
                if 'generated_ideas' in data:
                    ideas = data['generated_ideas']
                    if isinstance(ideas, list):
                        processed_ideas = []
                        for i, idea in enumerate(ideas):
                            # Handle different idea formats
                            if isinstance(idea, dict):
                                processed_ideas.append({
                                    'id': i,
                                    'title': idea.get('title', idea.get('idea_title', f'Idea {i+1}')),
                                    'status': 'Generated',
                                    'novelty_score': idea.get('novelty_score', idea.get('novelty', 'N/A')),
                                    'feasibility_score': idea.get('feasibility_score', idea.get('feasibility', 'N/A')),
                                    'description': idea.get('description', idea.get('summary', 'No description'))
                                })
                            else:
                                # Handle string ideas
                                processed_ideas.append({
                                    'id': i,
                                    'title': str(idea)[:100] + '...' if len(str(idea)) > 100 else str(idea),
                                    'status': 'Generated',
                                    'novelty_score': 'N/A',
                                    'feasibility_score': 'N/A',
                                    'description': str(idea)
                                })
                        session['generated_ideas'] = processed_ideas
                
                # Extract review results
                if 'review_results' in data:
                    session['review_results'] = data['review_results']
                
                # Extract final portfolio analysis
                if 'portfolio_analysis' in data:
                    session['portfolio_analysis'] = data['portfolio_analysis']
                
                # Store complete final output
                session['final_output'] = data
                        
        except Exception as e:
            # Log error but don't crash
            self._add_session_log(session_id, f"Data loading error: {str(e)[:100]}")
    
    def _cleanup_all_session_files(self):
        """Clean up all session files and directories before starting new research"""
        try:
            # Clear session manager internal state
            self.session_manager.sessions.clear()
            self.session_manager.completed_sessions.clear()
            
            # Clear UI internal state
            self.active_sessions.clear()
            self.current_session_id = None
            
            # Remove session files directory
            sessions_dir = Path('sessions')
            if sessions_dir.exists():
                import shutil
                shutil.rmtree(sessions_dir)
                sessions_dir.mkdir(exist_ok=True)
            
            # Clean up output files
            outputs_dir = Path('outputs')
            if outputs_dir.exists():
                for pattern in ['*.json', '*.txt', '*.summary.*']:
                    for output_file in outputs_dir.glob(pattern):
                        if output_file.name != 'README.md':  # Preserve README
                            try:
                                output_file.unlink()
                            except:
                                pass  # Continue if file can't be deleted
            
            # Clean up log files more thoroughly
            logs_dir = Path('logs')
            if logs_dir.exists():
                for pattern in ['*.log', '*.json', 'session_*']:
                    for log_file in logs_dir.glob(pattern):
                        try:
                            log_file.unlink()
                        except:
                            pass  # Continue if file can't be deleted

            # Clean up logs in unified structure
            logs_base = Path('logs')
            if logs_base.exists():
                for subdir in ['llm', 'session', 'idea', 'kg']:
                    subdir_path = logs_base / subdir
                    if subdir_path.exists():
                        for pattern in ['*.jsonl', '*.json', '*.log']:
                            for log_file in subdir_path.glob(pattern):
                                try:
                                    log_file.unlink()
                                except:
                                    pass  # Continue if file can't be deleted

        except Exception as e:
            # Log error but don't fail the session start
            print(f"Warning: Failed to cleanup session files: {e}")
    
    def _start_research_session(self, topic: str, num_ideas: int):
        """Start new research session using session manager"""
        if not topic.strip():
            return (
                "Error: Please enter research topic",
                "Error",
                "Please enter a valid research topic",
                {},
                "",
                gr.Dropdown(choices=[])
            )
        
        try:
            # Clean up all previous session files before starting new research
            self._cleanup_all_session_files()
            
            # Create session using session manager
            session_id = self.session_manager.create_session(
                topic=topic.strip(),
                num_ideas=num_ideas,
                debug_mode=False
            )
            
            # Start the session
            if self.session_manager.start_session(session_id):
                self.current_session_id = session_id
                
                # Also maintain legacy active_sessions for compatibility
                self.active_sessions[session_id] = {
                    'topic': topic.strip(),
                    'num_ideas': num_ideas,
                    'debug_mode': False,
                    'start_time': datetime.now(),
                    'status': 'running',
                    'logs': [],
                    'process': None,
                    'literature_papers': [],
                    'generated_ideas': [],
                    'review_results': {},
                    'current_phase': 'Initializing',
                    'progress': 0
                }
                
                # Update UI immediately and select the new session
                session_choices = self._refresh_sessions()

                # Find and select the newly created session in the dropdown
                new_session_value = None
                if hasattr(session_choices, 'choices'):
                    for choice in session_choices.choices:
                        if session_id in choice:
                            new_session_value = choice
                            break

                # Return with the new session selected
                if new_session_value:
                    session_choices = gr.Dropdown(choices=session_choices.choices, value=new_session_value)

                # Return session info with initial empty output/logs
                return (
                    f"Running - {session_id}",
                    "Initializing",
                    f"🚀 Research process launched: {topic} ({num_ideas} ideas)",
                    {},  # final_output_display (empty initially)
                    f"[{datetime.now().strftime('%H:%M:%S')}] Session {session_id} started\nTopic: {topic}\nNumber of ideas: {num_ideas}\n\nInitializing research pipeline...",  # logs_display
                    session_choices
                )
            else:
                return (
                    "Failed to start session",
                    "Error",
                    "Session start failed",
                    {},
                    "",
                    gr.Dropdown(choices=[])
                )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return (
                f"Start failed: {e}",
                "Error",
                f"Session start failed: {e}",
                {},
                f"Error: {str(e)}",
                gr.Dropdown(choices=[])
            )
    
    # Auto-refresh methods removed - now using Gradio's built-in Timer component
    
    def _refresh_sessions(self):
        """Refresh session list (active sessions only)"""
        choices = []

        # Add active sessions from session manager (primary source)
        for session_id in self.session_manager.list_active_sessions():
            session = self.session_manager.sessions.get(session_id, {})
            topic = session.get('topic', 'Unknown')
            status = session.get('status', 'unknown')
            choices.append(f"🟢 {session_id}: {topic} ({status})")

        # Add any remaining legacy active sessions not in session manager
        for session_id in self.active_sessions.keys():
            if session_id not in self.session_manager.sessions:
                session = self.active_sessions[session_id]
                topic = session.get('topic', 'Unknown')
                status = session.get('status', 'unknown')
                choices.append(f"🟡 {session_id}: {topic} ({status}, legacy)")

        return gr.Dropdown(choices=choices)

    def _switch_session(self, session_selection: str):
        """Switch to specified session"""
        if not session_selection:
            return "No session", "Idle", "Select a session", {}, ""
        
        # Extract session ID from the dropdown format "🟢 session_id: topic (status)"
        try:
            session_id = session_selection.split(': ')[0].split(' ', 1)[1]
        except (IndexError, AttributeError):
            session_id = session_selection
        
        self.current_session_id = session_id
        
        # Check active sessions first
        if session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
        # Then check session manager for both active and completed
        elif session_id in self.session_manager.sessions:
            session_info = self.session_manager.sessions[session_id]
        elif session_id in self.session_manager.completed_sessions:
            session_info = self.session_manager.completed_sessions[session_id]
        else:
            return "Session not found", "Error", "Session does not exist", {}, ""
        
        # Load real-time data from session manager results
        session_results = self.session_manager.get_session_results(session_id)
        literature_df = session_results.get('literature', pd.DataFrame())
        ideas_df = session_results.get('ideas', pd.DataFrame())
        
        # If no session manager data, fall back to active_sessions data
        if literature_df.empty and session_id in self.active_sessions:
            papers = self.active_sessions[session_id].get('literature_papers', [])
            if papers:
                lit_data = []
                for p in papers:
                    authors_str = ', '.join(p.get('authors', [])) if isinstance(p.get('authors'), list) else str(p.get('authors', ''))
                    lit_data.append([
                        p.get('title', 'Unknown'),
                        authors_str,
                        p.get('year', 'N/A'),
                        p.get('citationCount', 'N/A')
                    ])
                literature_df = pd.DataFrame(lit_data, columns=['Title', 'Authors', 'Year', 'Citations'])
        
        # If no session manager data, fall back to active_sessions ideas
        if ideas_df.empty and session_id in self.active_sessions:
            ideas = self.active_sessions[session_id].get('generated_ideas', [])
            if ideas:
                ideas_data = []
                for i, idea in enumerate(ideas):
                    ideas_data.append([
                        idea.get('id', i),
                        idea.get('title', 'Untitled'),
                        idea.get('status', 'Generating...'),
                        idea.get('novelty_score', 'N/A'),
                        idea.get('feasibility_score', 'N/A')
                    ])
                ideas_df = pd.DataFrame(ideas_data, columns=['ID', 'Title', 'Status', 'Novelty', 'Feasibility'])
        
        # Get session logs from session manager with debugging
        try:
            logs_list = self.session_manager.get_session_logs(session_id, limit=50)
            if logs_list:
                logs_text = "\n".join(logs_list)
            else:
                logs_text = f"[DEBUG] Session {session_id} has no logs yet\n[DEBUG] Active sessions: {list(self.session_manager.sessions.keys())}"
        except Exception as e:
            logs_text = f"[ERROR] Failed to get logs: {e}"
        
        # Get session status from session manager
        session_status = self.session_manager.get_session_status(session_id)
        if session_status:
            current_phase = session_status.get('phase', 'Research in Progress')
            topic = session_status.get('topic', 'Unknown topic')
        else:
            current_phase = session_info.get('current_phase', 'Research in Progress')
            topic = session_info.get('topic', 'Unknown topic')
        
        # Get review results and final output
        review_results = session_results.get('review_results', {})
        final_output = session_results.get('final_output', {})
        
        # Format status based on session state
        if session_status:
            status_text = f"{session_status.get('status', 'unknown').title()} - {session_id}"
            if session_status.get('status') == 'completed':
                duration = session_status.get('duration', 0)
                status_text += f" (completed in {duration:.0f}s)"
        else:
            status_text = f"Active - {session_id}"
        
        # For completed sessions, add completion info to logs
        if session_status and session_status.get('status') == 'completed' and 'end_time' in session_status:
            completion_time = session_status['end_time']
            logs_text += f"\n\n=== Session Completed at {completion_time} ==="
        
        return (
            status_text,
            current_phase,
            f"Topic: {topic}",
            final_output,
            logs_text
        )
    
    def _terminate_session(self, session_selection: str):
        """Terminate specified session"""
        if not session_selection:
            return (
                "No session",
                "Idle",
                "No session selected",
                {},
                "",
                self._refresh_sessions()
            )

        # Extract session ID from dropdown selection
        try:
            session_id = session_selection.split(': ')[0].split(' ', 1)[1]
        except (IndexError, AttributeError):
            session_id = session_selection

        try:
            # Get current session data before terminating (to preserve output)
            session_data = None
            if session_id in self.session_manager.sessions:
                session_results = self.session_manager.get_session_results(session_id)
                session_status = self.session_manager.get_session_status(session_id)
                logs_list = self.session_manager.get_session_logs(session_id, limit=50)

                topic = session_status.get('topic', 'Unknown') if session_status else 'Unknown'
                phase = session_status.get('phase', 'Terminated') if session_status else 'Terminated'
                final_output = session_results.get('final_output', {})
                logs_text = "\n".join(logs_list) if logs_list else ""

                session_data = (
                    f"Terminated - {session_id}",
                    phase,
                    f"Topic: {topic} (Session terminated)",
                    final_output,
                    logs_text + "\n\n=== Session Terminated ==="
                )

            # Try to terminate via session manager
            success = self.session_manager.terminate_session(session_id)

            # Also clean up from active_sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

            if self.current_session_id == session_id:
                self.current_session_id = None

            if success:
                if session_data:
                    # Return preserved session data
                    return session_data + (self._refresh_sessions(),)
                else:
                    return (
                        "Session terminated",
                        "Idle",
                        "Session terminated",
                        {},
                        "Session terminated successfully",
                        self._refresh_sessions()
                    )
            else:
                return (
                    "Session not found or already terminated",
                    "Idle",
                    "Session not found",
                    {},
                    "",
                    self._refresh_sessions()
                )

        except Exception as e:
            return (
                f"Termination failed: {e}",
                "Error",
                f"Failed to terminate: {e}",
                {},
                "",
                self._refresh_sessions()
            )
    
    def _sync_session_data(self, session_id: str):
        """Sync data between active_sessions and session_manager"""
        if session_id in self.active_sessions and session_id in self.session_manager.sessions:
            active_data = self.active_sessions[session_id]
            manager_data = self.session_manager.sessions[session_id]
            
            # Sync literature papers
            if active_data.get('literature_papers') and not manager_data.get('literature_papers'):
                manager_data['literature_papers'] = active_data['literature_papers']
            
            # Sync generated ideas
            if active_data.get('generated_ideas') and not manager_data.get('generated_ideas'):
                manager_data['generated_ideas'] = active_data['generated_ideas']
            
            # Sync review results
            if active_data.get('review_results') and not manager_data.get('review_results'):
                manager_data['review_results'] = active_data['review_results']
    
    
    def _auto_refresh_display(self):
        """Auto-refresh display using internal session tracking"""
        if not self.current_session_id:
            return (
                "No active session",
                "Idle",
                "Start a research session to begin",
                {},
                "[No active session - start a research session to see logs]"
            )

        # Get session status and data
        session_status = self.session_manager.get_session_status(self.current_session_id)

        # Use the internal session ID instead of dropdown value
        final_output, logs = self._refresh_session_data(self.current_session_id)

        # Get current phase and status text
        if session_status:
            status = session_status.get('status', 'unknown')
            phase = session_status.get('phase', 'Unknown')
            topic = session_status.get('topic', 'Unknown')

            # Format status text
            if status == 'completed':
                duration = session_status.get('duration', 0)
                status_text = f"Completed - {self.current_session_id} (in {duration:.0f}s)"
            elif status == 'running':
                status_text = f"Running - {self.current_session_id}"
            else:
                status_text = f"{status.title()} - {self.current_session_id}"

            progress_text = f"Topic: {topic}"
        else:
            status_text = f"Active - {self.current_session_id}"
            phase = "Research in Progress"
            progress_text = "Running..."

        return (
            status_text,
            phase,
            progress_text,
            final_output,
            logs
        )
    
    def _refresh_session_data(self, session_id: str):
        """Core session data refresh logic"""
        if not session_id:
            empty_df = pd.DataFrame()
            return empty_df, empty_df, {}, {}, "[No session ID provided]"
        
        # Check both active and completed sessions
        if (session_id not in self.active_sessions and 
            session_id not in self.session_manager.sessions and 
            session_id not in self.session_manager.completed_sessions):
            empty_df = pd.DataFrame()
            return empty_df, empty_df, {}, {}, f"[Session {session_id} not found - may have been cleaned up]"
        
        # Load real data from session manager
        session_results = self.session_manager.get_session_results(session_id)
        
        # Also try to load from output files for active sessions
        if session_id in self.active_sessions:
            self._load_real_output_data(session_id)
        
        # Get DataFrames from session manager
        literature_df = session_results.get('literature', pd.DataFrame())
        ideas_df = session_results.get('ideas', pd.DataFrame())
        
        # Fall back to active session data if needed
        if literature_df.empty and session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            papers = session_info.get('literature_papers', [])
            if papers:
                lit_data = []
                for p in papers:
                    # Handle both dict and string formats for authors
                    if isinstance(p.get('authors'), list):
                        authors_str = ', '.join(p.get('authors', [])[:3])  # Show first 3 authors
                        if len(p.get('authors', [])) > 3:
                            authors_str += ' et al.'
                    else:
                        authors_str = str(p.get('authors', 'Unknown'))
                    
                    lit_data.append([
                        p.get('title', 'Unknown')[:100],  # Truncate long titles
                        authors_str[:50],  # Truncate long author lists
                        str(p.get('year', 'N/A')),
                        str(p.get('citationCount', 0))
                    ])
                literature_df = pd.DataFrame(lit_data, columns=['Title', 'Authors', 'Year', 'Citations'])
        
        # Fall back to active session ideas if needed
        if ideas_df.empty and session_id in self.active_sessions:
            session_info = self.active_sessions[session_id]
            ideas = session_info.get('generated_ideas', [])
            if ideas:
                ideas_data = []
                for i, idea in enumerate(ideas):
                    # Extract scores if available
                    novelty = idea.get('novelty_score', 'N/A')
                    feasibility = idea.get('feasibility_score', 'N/A')
                    
                    # Format scores if they're numbers
                    if isinstance(novelty, (int, float)):
                        novelty = f"{novelty:.2f}"
                    if isinstance(feasibility, (int, float)):
                        feasibility = f"{feasibility:.2f}"
                        
                    ideas_data.append([
                        idea.get('id', i),
                        idea.get('title', f'Idea {i+1}')[:100],  # Truncate long titles
                        idea.get('status', 'Generating...'),
                        novelty,
                        feasibility
                    ])
                ideas_df = pd.DataFrame(ideas_data, columns=['ID', 'Title', 'Status', 'Novelty', 'Feasibility'])
        
        # Get session logs from session manager with debugging info
        try:
            logs_list = self.session_manager.get_session_logs(session_id, limit=50)
            if logs_list:
                logs_text = "\n".join(logs_list)
            else:
                logs_text = f"[DEBUG] No logs returned for session {session_id}\n[DEBUG] Session exists in manager: {session_id in self.session_manager.sessions}"
        except Exception as e:
            logs_text = f"[ERROR] Failed to get logs for {session_id}: {e}"
        
        # Get final output from session results with better handling
        final_output = session_results.get('final_output', {})
        
        # If no final output from session manager, try to load directly from files
        if not final_output:
            final_output = self._try_load_final_output(session_id)
        
        # Add debug info if still no final output
        if not final_output:
            status = self.session_manager.get_session_status(session_id)
            if status and status.get('status') == 'completed':
                final_output = {
                    'status': 'Research completed but no output file found',
                    'debug_info': f'Session {session_id} completed but final output not available'
                }
        
        return final_output, logs_text
    
    def _try_load_final_output(self, session_id: str) -> dict:
        """Try to load final output directly from output files"""
        try:
            # Get session info to find topic
            session = self.session_manager.sessions.get(session_id) or self.session_manager.completed_sessions.get(session_id)
            if not session:
                return {}
            
            topic = session.get('topic', '')
            if not topic:
                return {}
            
            # Look for output files matching the topic
            outputs_dir = Path('outputs')
            if not outputs_dir.exists():
                return {}
            
            # Try different patterns to find the output file
            topic_pattern = topic.replace(' ', '_').lower()
            
            # Find matching files (try multiple patterns)
            matching_files = []
            
            # Pattern 1: topic_timestamp.json
            for json_file in outputs_dir.glob('*.json'):
                if topic_pattern in json_file.name.lower():
                    matching_files.append(json_file)
            
            # Pattern 2: recent files (within last 30 minutes)
            if not matching_files:
                for json_file in outputs_dir.glob('*.json'):
                    if time.time() - json_file.stat().st_mtime < 1800:  # 30 minutes
                        matching_files.append(json_file)
            
            if matching_files:
                # Get the most recent file
                latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        except Exception as e:
            return {'error': f'Failed to load output file: {e}'}
        
        return {}
    
    
    def _auto_refresh_sessions(self):
        """Auto-refresh session list while preserving current selection"""
        # Get updated session choices from the method
        dropdown_update = self._refresh_sessions()

        # Find the choice that matches our current session if we have one
        selected_value = None
        if self.current_session_id:
            choices = dropdown_update.choices if hasattr(dropdown_update, 'choices') else []
            for choice in choices:
                if self.current_session_id in choice:
                    selected_value = choice
                    break

        # Return dropdown with proper selection
        if selected_value:
            return gr.Dropdown(choices=dropdown_update.choices, value=selected_value)
        else:
            return dropdown_update
    
    
    
    def launch(self):
        """Launch the process visualization interface"""
        if not self.interface:
            self.interface = self.create_interface()
        
        # Try to find available port if specified port is occupied
        import socket
        
        def is_port_available(port):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind((self.host, port))
                return True
            except OSError:
                return False
        
        # Find available port starting from self.port
        available_port = self.port
        for port_offset in range(10):  # Try 10 ports
            test_port = self.port + port_offset
            if is_port_available(test_port):
                available_port = test_port
                break
        
        if available_port != self.port:
            print(f"⚠️  Port {self.port} occupied, using port {available_port} instead")
        
        print(f"🌐 Starting UI at: http://{self.host}:{available_port}")
        
        self.interface.launch(
            server_name=self.host,
            server_port=available_port,
            share=self.share,
            show_error=True,
            inbrowser=True
        )


