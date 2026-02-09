#!/usr/bin/env python3
"""
Pre-Generation Validation Module

This module validates all external dependencies and services before running 
the main research agent workflow. It checks:
- LLM endpoint connectivity and response
- Semantic Scholar API accessibility  
- Retrieval model availability and functionality
- Configuration validity

This ensures all required services are working before starting expensive operations.
"""

import requests
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from .debug_logger import DebugLogger


@dataclass
class ValidationResult:
    """Result of a validation test"""
    service: str
    test_name: str
    success: bool
    message: str
    response_time_ms: Optional[float] = None
    details: Optional[Dict] = None


class PregenValidator:
    """Validates external dependencies before main workflow execution"""
    
    def __init__(self, config: Dict, logger: DebugLogger):
        self.config = config
        self.logger = logger
        self.results: List[ValidationResult] = []
        
    def validate_all(self) -> Tuple[bool, List[ValidationResult]]:
        """Run all validation tests"""
        self.logger.log_info("Starting pre-generation validation", "pregen_validator")
        
        # Clear previous results
        self.results = []
        
        # Run all validations
        self._validate_llm_endpoint()
        self._validate_semantic_scholar_api()
        self._validate_retrieval_models()
        self._validate_configuration()
        
        # Calculate overall success
        all_passed = all(result.success for result in self.results)
        
        self.logger.log_info(f"Pre-generation validation completed. Success: {all_passed}", "pregen_validator")
        return all_passed, self.results
    
    def _validate_llm_endpoint(self):
        """Validate LLM endpoint connectivity and response"""
        self.logger.log_info("Validating LLM endpoint", "pregen_validator")
        
        llm_config = self.config.get('llm', {})
        
        # Check configuration exists
        if not llm_config:
            self.results.append(ValidationResult(
                service="LLM",
                test_name="Configuration Check",
                success=False,
                message="LLM configuration not found"
            ))
            return
        
        # Check required fields
        required_fields = ['api_key', 'base_url', 'model_name']
        missing_fields = [field for field in required_fields if not llm_config.get(field)]
        
        if missing_fields:
            self.results.append(ValidationResult(
                service="LLM",
                test_name="Configuration Check", 
                success=False,
                message=f"Missing required LLM config fields: {', '.join(missing_fields)}"
            ))
            return
        
        self.results.append(ValidationResult(
            service="LLM",
            test_name="Configuration Check",
            success=True,
            message="LLM configuration is valid"
        ))
        
        # Test endpoint connectivity
        self._test_llm_endpoint_connectivity(llm_config)
        
        # Test model response
        self._test_llm_model_response(llm_config)
    
    def _test_llm_endpoint_connectivity(self, llm_config: Dict):
        """Test basic connectivity to LLM endpoint"""
        base_url = llm_config['base_url']
        api_key = llm_config['api_key']
        
        try:
            start_time = time.time()
            
            # Test models endpoint if available
            models_url = f"{base_url}/models"
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(models_url, headers=headers, timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.results.append(ValidationResult(
                    service="LLM",
                    test_name="Endpoint Connectivity",
                    success=True,
                    message=f"LLM endpoint is accessible",
                    response_time_ms=response_time
                ))
            else:
                self.results.append(ValidationResult(
                    service="LLM",
                    test_name="Endpoint Connectivity",
                    success=False,
                    message=f"LLM endpoint returned {response.status_code}: {response.text[:100]}"
                ))
                
        except requests.exceptions.Timeout:
            self.results.append(ValidationResult(
                service="LLM",
                test_name="Endpoint Connectivity",
                success=False,
                message="LLM endpoint timed out (>10s)"
            ))
        except requests.exceptions.ConnectionError:
            self.results.append(ValidationResult(
                service="LLM", 
                test_name="Endpoint Connectivity",
                success=False,
                message="Cannot connect to LLM endpoint"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                service="LLM",
                test_name="Endpoint Connectivity", 
                success=False,
                message=f"LLM connectivity test failed: {str(e)}"
            ))
    
    def _test_llm_model_response(self, llm_config: Dict):
        """Test LLM model can generate responses"""
        base_url = llm_config['base_url']
        api_key = llm_config['api_key']
        model_name = llm_config['model_name']
        
        try:
            start_time = time.time()
            
            # Simple test prompt
            test_payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'validation test successful' if you can respond."}
                ],
                "max_tokens": 1024  # Increased for reasoning models that use reasoning tokens
                # Note: temperature removed - Gemini models don't work well with low temperature values
            }
            
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=60  # Increased timeout for models with reasoning (e.g., Gemini)
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'choices' in data and data['choices']:
                    content = data['choices'][0].get('message', {}).get('content', '')
                    self.results.append(ValidationResult(
                        service="LLM",
                        test_name="Model Response",
                        success=True,
                        message=f"LLM model responded successfully",
                        response_time_ms=response_time,
                        details={"response_preview": content[:50]}
                    ))
                else:
                    self.results.append(ValidationResult(
                        service="LLM",
                        test_name="Model Response",
                        success=False,
                        message="LLM response missing expected structure"
                    ))
            else:
                self.results.append(ValidationResult(
                    service="LLM",
                    test_name="Model Response",
                    success=False,
                    message=f"LLM model test failed: {response.status_code} - {response.text[:100]}"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                service="LLM",
                test_name="Model Response",
                success=False,
                message=f"LLM model test failed: {str(e)}"
            ))
    
    def _validate_semantic_scholar_api(self):
        """Validate Semantic Scholar API accessibility"""
        self.logger.log_info("Validating Semantic Scholar API", "pregen_validator")
        
        scholar_config = self.config.get('semantic_scholar', {})
        
        # Check configuration
        if not scholar_config or not scholar_config.get('api_key'):
            self.results.append(ValidationResult(
                service="Semantic Scholar",
                test_name="Configuration Check",
                success=False,
                message="Semantic Scholar API key not configured"
            ))
            return
        
        self.results.append(ValidationResult(
            service="Semantic Scholar",
            test_name="Configuration Check",
            success=True,
            message="Semantic Scholar API configuration found"
        ))
        
        # Test API connectivity
        self._test_semantic_scholar_connectivity(scholar_config)
        
        # Test search functionality
        self._test_semantic_scholar_search(scholar_config)
    
    def _test_semantic_scholar_connectivity(self, scholar_config: Dict):
        """Test Semantic Scholar API connectivity"""
        api_key = scholar_config['api_key']
        base_url = "https://api.semanticscholar.org/graph/v1"
        
        try:
            start_time = time.time()
            
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json'
            }
            
            # Test with a simple paper lookup
            test_url = f"{base_url}/paper/649def34f8be52c8b66281af98ae884c09aef38b"
            response = requests.get(test_url, headers=headers, timeout=10)
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                self.results.append(ValidationResult(
                    service="Semantic Scholar",
                    test_name="API Connectivity",
                    success=True,
                    message="Semantic Scholar API is accessible",
                    response_time_ms=response_time
                ))
            elif response.status_code == 429:
                self.results.append(ValidationResult(
                    service="Semantic Scholar",
                    test_name="API Connectivity",
                    success=True,  # Rate limiting means API is working
                    message="Semantic Scholar API rate limited - system will handle retries"
                ))
            else:
                self.results.append(ValidationResult(
                    service="Semantic Scholar",
                    test_name="API Connectivity",
                    success=False,
                    message=f"Semantic Scholar API returned {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                service="Semantic Scholar",
                test_name="API Connectivity",
                success=False,
                message=f"Semantic Scholar connectivity failed: {str(e)}"
            ))
    
    def _test_semantic_scholar_search(self, scholar_config: Dict):
        """Test Semantic Scholar search functionality"""
        api_key = scholar_config['api_key']
        base_url = "https://api.semanticscholar.org/graph/v1"
        
        try:
            start_time = time.time()
            
            headers = {
                'x-api-key': api_key,
                'Content-Type': 'application/json'
            }
            
            # Test search with simple query
            params = {
                'query': 'machine learning',
                'limit': 3,
                'fields': 'paperId,title,authors,year'
            }
            
            response = requests.get(
                f"{base_url}/paper/search",
                headers=headers,
                params=params,
                timeout=15
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    paper_count = len(data['data'])
                    self.results.append(ValidationResult(
                        service="Semantic Scholar",
                        test_name="Search Functionality",
                        success=True,
                        message=f"Search returned {paper_count} papers",
                        response_time_ms=response_time
                    ))
                else:
                    self.results.append(ValidationResult(
                        service="Semantic Scholar",
                        test_name="Search Functionality",
                        success=False,
                        message="Search returned no results"
                    ))
            elif response.status_code == 429:
                self.results.append(ValidationResult(
                    service="Semantic Scholar",
                    test_name="Search Functionality",
                    success=True,  # Treat rate limiting as success since it means API is working
                    message="Search rate limited - system will handle retries during execution"
                ))
            else:
                self.results.append(ValidationResult(
                    service="Semantic Scholar",
                    test_name="Search Functionality",
                    success=False,
                    message=f"Search failed: {response.status_code}"
                ))
                
        except Exception as e:
            self.results.append(ValidationResult(
                service="Semantic Scholar",
                test_name="Search Functionality",
                success=False,
                message=f"Search test failed: {str(e)}"
            ))
    
    def _validate_retrieval_models(self):
        """Validate retrieval models are available"""
        self.logger.log_info("Validating retrieval models", "pregen_validator")
        
        # Test TF-IDF availability (always available with scikit-learn)
        self._test_tfidf_availability()
        
        # Test BGE model availability (optional)
        self._test_bge_model_availability()
    
    def _test_tfidf_availability(self):
        """Test TF-IDF vectorization availability"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Quick test
            vectorizer = TfidfVectorizer()
            test_docs = ["machine learning", "artificial intelligence", "deep learning"]
            vectors = vectorizer.fit_transform(test_docs)
            similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            
            if len(similarity) > 0:
                self.results.append(ValidationResult(
                    service="Retrieval Models",
                    test_name="TF-IDF Vectorization",
                    success=True,
                    message="TF-IDF vectorization working correctly"
                ))
            else:
                self.results.append(ValidationResult(
                    service="Retrieval Models",
                    test_name="TF-IDF Vectorization",
                    success=False,
                    message="TF-IDF test returned empty results"
                ))
                
        except ImportError as e:
            self.results.append(ValidationResult(
                service="Retrieval Models",
                test_name="TF-IDF Vectorization",
                success=False,
                message=f"TF-IDF dependencies missing: {str(e)}"
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                service="Retrieval Models",
                test_name="TF-IDF Vectorization",
                success=False,
                message=f"TF-IDF test failed: {str(e)}"
            ))
    
    def _test_bge_model_availability(self):
        """Test BGE embedding model availability (optional)"""
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try to load a small BGE model
            start_time = time.time()
            model = SentenceTransformer('BAAI/bge-small-en-v1.5')
            load_time = (time.time() - start_time) * 1000
            
            # Quick encoding test
            test_texts = ["machine learning", "artificial intelligence"]
            embeddings = model.encode(test_texts)
            
            if embeddings is not None and len(embeddings) == 2:
                self.results.append(ValidationResult(
                    service="Retrieval Models",
                    test_name="BGE Model",
                    success=True,
                    message="BGE embedding model working correctly",
                    response_time_ms=load_time
                ))
            else:
                self.results.append(ValidationResult(
                    service="Retrieval Models", 
                    test_name="BGE Model",
                    success=False,
                    message="BGE model test returned invalid results"
                ))
                
        except ImportError:
            self.results.append(ValidationResult(
                service="Retrieval Models",
                test_name="BGE Model", 
                success=False,
                message="BGE model not available (sentence-transformers not installed). Using TF-IDF fallback."
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                service="Retrieval Models",
                test_name="BGE Model",
                success=False,
                message=f"BGE model test failed: {str(e)}. Using TF-IDF fallback."
            ))
    
    def _validate_configuration(self):
        """Validate overall configuration integrity"""
        self.logger.log_info("Validating configuration integrity", "pregen_validator")
        
        # Check all required top-level sections
        required_sections = ['llm', 'semantic_scholar', 'external_selector']
        missing_sections = [section for section in required_sections if section not in self.config]
        
        if missing_sections:
            self.results.append(ValidationResult(
                service="Configuration",
                test_name="Section Completeness",
                success=False,
                message=f"Missing configuration sections: {', '.join(missing_sections)}"
            ))
        else:
            self.results.append(ValidationResult(
                service="Configuration",
                test_name="Section Completeness",
                success=True,
                message="All required configuration sections present"
            ))
        
        # Validate configuration values
        self._validate_config_values()
    
    def _validate_config_values(self):
        """Validate configuration value ranges and types"""
        issues = []
        
        # Check external selector config
        external_selector_config = self.config.get('external_selector', {})
        threshold = external_selector_config.get('similarity_threshold', 0.7)
        if not 0 <= threshold <= 1:
            issues.append("external_selector.similarity_threshold must be between 0 and 1")
        
        # Check semantic scholar config
        scholar_config = self.config.get('semantic_scholar', {})
        num_papers = scholar_config.get('num_papers', 10)
        if not isinstance(num_papers, int) or num_papers < 1:
            issues.append("semantic_scholar.num_papers must be a positive integer")
        
        # Check LLM config
        llm_config = self.config.get('llm', {})
        max_tokens = llm_config.get('max_tokens', 500)
        if not isinstance(max_tokens, int) or max_tokens < 1:
            issues.append("llm.max_tokens must be a positive integer")
        
        temperature = llm_config.get('temperature', 0.7)
        if not 0 <= temperature <= 2:
            issues.append("llm.temperature must be between 0 and 2")
        
        if issues:
            self.results.append(ValidationResult(
                service="Configuration",
                test_name="Value Validation",
                success=False,
                message=f"Configuration issues: {'; '.join(issues)}"
            ))
        else:
            self.results.append(ValidationResult(
                service="Configuration",
                test_name="Value Validation",
                success=True,
                message="Configuration values are valid"
            ))
    
    def print_validation_report(self):
        """Print a comprehensive validation report"""
        if not self.results:
            print("❌ No validation results available")
            return
        
        print("\n" + "="*70)
        print("🔍 PRE-GENERATION VALIDATION REPORT")
        print("="*70)
        
        # Group results by service
        services = {}
        for result in self.results:
            if result.service not in services:
                services[result.service] = []
            services[result.service].append(result)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.success)
        
        # Print summary
        print(f"\n📊 SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        # Print each service
        for service, results in services.items():
            service_passed = sum(1 for r in results if r.success)
            service_total = len(results)
            
            print(f"\n🔧 {service} ({service_passed}/{service_total})")
            print("-" * 40)
            
            for result in results:
                status = "✅" if result.success else "❌"
                timing = f" ({result.response_time_ms:.0f}ms)" if result.response_time_ms else ""
                print(f"  {status} {result.test_name}: {result.message}{timing}")
                
                if result.details:
                    for key, value in result.details.items():
                        print(f"     💡 {key}: {value}")
        
        print("\n" + "="*70)
        
        # Overall result
        overall_success = passed_tests == total_tests
        if overall_success:
            print("🎉 All validation tests passed! System is ready.")
        else:
            print("⚠️  Some validation tests failed. Check the issues above.")
        
        print("="*70)
        
        return overall_success


def run_pregen_validation(config: Dict, logger: DebugLogger) -> Tuple[bool, List[ValidationResult]]:
    """
    Run pre-generation validation tests
    
    Args:
        config: System configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (all_passed, validation_results)
    """
    validator = PregenValidator(config, logger)
    return validator.validate_all()


def print_validation_report(results: List[ValidationResult]) -> bool:
    """
    Print validation report and return overall success
    
    Args:
        results: List of validation results
        
    Returns:
        True if all tests passed, False otherwise
    """
    validator = PregenValidator({}, None)
    validator.results = results
    return validator.print_validation_report()


if __name__ == "__main__":
    # Standalone test mode
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from utils.config import load_config
    from utils.debug_logger import init_debug_logger
    
    # Load config and run validation
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'agent_config.yaml')
    config = load_config(config_path)
    logger = init_debug_logger(debug_mode=True)
    
    success, results = run_pregen_validation(config, logger)
    print_validation_report(results)