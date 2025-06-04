"""
Response generation module for creating concise legal responses.
Uses OpenAI ChatGPT with retrieved document context to generate brief, actionable responses.
"""

import asyncio
import time
from typing import Dict, List, Optional
from openai import AsyncOpenAI
from rich.console import Console

console = Console()

class LegalResponder:
    """
    Legal response generator that creates concise, actionable responses
    for courtroom use based on retrieved document context.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", 
                 temperature: float = 0.2, max_tokens: int = 160):
        """
        Initialize the legal responder.
        
        Args:
            api_key: OpenAI API key
            model: ChatGPT model to use
            temperature: Response randomness (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum response length in tokens
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Track response generation statistics
        self.total_requests = 0
        self.failed_requests = 0
        self.total_response_time = 0
        
        console.print(f"[green]Legal responder initialized:[/green] {model}, temp={temperature}, max_tokens={max_tokens}")
    
    def create_system_prompt(self) -> str:
        """
        Create the system prompt that defines the AI's role and response format.
        
        Returns:
            System prompt string
        """
        return """You are a skilled legal assistant providing real-time support during courtroom proceedings. Your role is to:

1. Analyze incoming speech from court participants (questions, claims, objections, citation requests)
2. Use provided legal document context to craft precise, actionable responses
3. Generate responses that can be quickly read aloud or whispered to counsel

RESPONSE REQUIREMENTS:
- Maximum 3 sentences
- Focus on the most critical legal point or counter-argument
- Include specific case names, statutes, or rules when available
- Use confident, professional language suitable for court
- If no relevant context is found, briefly state "No direct precedent found" and suggest a general legal principle

RESPONSE TYPES:
- For QUESTIONS: Provide direct, factual answers with citations
- For CLAIMS: Offer counter-arguments or supporting precedent
- For CITATION REQUESTS: Provide specific case names, statutes, or rules
- For OBJECTIONS: Suggest appropriate responses or legal basis

Keep responses concise, actionable, and immediately useful in a live courtroom setting."""

    def create_user_prompt(self, speech_text: str, context: str, speech_patterns: Dict) -> str:
        """
        Create the user prompt with speech text, context, and detected patterns.
        
        Args:
            speech_text: Transcribed speech from courtroom
            context: Retrieved document context
            speech_patterns: Detected speech patterns (questions, claims, etc.)
            
        Returns:
            Formatted user prompt
        """
        # Determine the type of response needed
        if speech_patterns.get("is_question"):
            response_type = "QUESTION"
        elif speech_patterns.get("is_citation_request"):
            response_type = "CITATION REQUEST"
        elif speech_patterns.get("is_claim"):
            response_type = "CLAIM"
        elif speech_patterns.get("is_objection"):
            response_type = "OBJECTION"
        else:
            response_type = "GENERAL"
        
        prompt = f"""SPEECH TYPE: {response_type}

TRANSCRIBED SPEECH:
"{speech_text}"

RELEVANT LEGAL CONTEXT:
{context}

Generate a concise, actionable response (max 3 sentences) that addresses the speech and incorporates the legal context. Focus on providing immediate value for courtroom use."""
        
        return prompt
    
    async def generate_response(self, speech_text: str, context: str, 
                              speech_patterns: Dict, timeout: float = 4.0) -> Dict:
        """
        Generate a legal response based on speech text and retrieved context.
        
        Args:
            speech_text: Transcribed speech from courtroom
            context: Retrieved document context
            speech_patterns: Detected speech patterns
            timeout: Maximum time to wait for response (seconds)
            
        Returns:
            Response dictionary with generated text and metadata
        """
        start_time = time.time()
        
        try:
            # Create prompts
            system_prompt = self.create_system_prompt()
            user_prompt = self.create_user_prompt(speech_text, context, speech_patterns)
            
            # Generate response with timeout
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=False
                ),
                timeout=timeout
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Update statistics
            self.total_requests += 1
            self.total_response_time += processing_time
            
            result = {
                "response_text": response_text,
                "processing_time": processing_time,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "success": True,
                "speech_type": self._determine_speech_type(speech_patterns),
                "has_context": bool(context.strip())
            }
            
            # Log successful response
            console.print(f"[green]💬 Response generated:[/green] '{response_text[:80]}{'...' if len(response_text) > 80 else ''}' ({processing_time:.2f}s)")
            
            return result
            
        except asyncio.TimeoutError:
            console.print(f"[red]Response generation timed out after {timeout}s[/red]")
            self.failed_requests += 1
            
            return {
                "response_text": "Response timeout - unable to generate answer within time limit.",
                "processing_time": timeout,
                "model": self.model,
                "tokens_used": 0,
                "success": False,
                "error": "timeout",
                "speech_type": self._determine_speech_type(speech_patterns),
                "has_context": bool(context.strip())
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            console.print(f"[red]Response generation failed: {e}[/red]")
            self.failed_requests += 1
            
            return {
                "response_text": f"Error generating response: {str(e)[:100]}",
                "processing_time": processing_time,
                "model": self.model,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
                "speech_type": self._determine_speech_type(speech_patterns),
                "has_context": bool(context.strip())
            }
    
    def _determine_speech_type(self, speech_patterns: Dict) -> str:
        """Determine the primary speech type from patterns."""
        if speech_patterns.get("is_objection"):
            return "objection"
        elif speech_patterns.get("is_citation_request"):
            return "citation_request"
        elif speech_patterns.get("is_question"):
            return "question"
        elif speech_patterns.get("is_claim"):
            return "claim"
        else:
            return "general"
    
    async def generate_quick_response(self, speech_text: str, max_time: float = 2.0) -> Dict:
        """
        Generate a quick response without document context for time-critical situations.
        
        Args:
            speech_text: Transcribed speech
            max_time: Maximum response time
            
        Returns:
            Quick response dictionary
        """
        start_time = time.time()
        
        try:
            # Simplified prompt for quick responses
            quick_prompt = f"""Provide a brief legal response to this courtroom statement (max 2 sentences):

"{speech_text}"

Focus on the most important legal principle or procedure that applies."""
            
            response = await asyncio.wait_for(
                self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a legal assistant providing quick courtroom guidance."},
                        {"role": "user", "content": quick_prompt}
                    ],
                    temperature=0.1,  # Lower temperature for quick responses
                    max_tokens=80,    # Shorter responses
                    stream=False
                ),
                timeout=max_time
            )
            
            processing_time = time.time() - start_time
            response_text = response.choices[0].message.content.strip()
            
            console.print(f"[yellow]⚡ Quick response:[/yellow] '{response_text}' ({processing_time:.2f}s)")
            
            return {
                "response_text": response_text,
                "processing_time": processing_time,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "success": True,
                "response_type": "quick"
            }
            
        except Exception as e:
            return {
                "response_text": "Unable to generate quick response.",
                "processing_time": time.time() - start_time,
                "model": self.model,
                "tokens_used": 0,
                "success": False,
                "error": str(e),
                "response_type": "quick"
            }
    
    def format_response_for_display(self, response_dict: Dict, speech_text: str = "") -> str:
        """
        Format the response for console display.
        
        Args:
            response_dict: Response dictionary from generate_response()
            speech_text: Original speech text for context
            
        Returns:
            Formatted display string
        """
        if not response_dict["success"]:
            return f"[red]❌ Error:[/red] {response_dict.get('error', 'Unknown error')}"
        
        response_text = response_dict["response_text"]
        processing_time = response_dict["processing_time"]
        speech_type = response_dict.get("speech_type", "unknown")
        
        # Add response type indicator
        type_indicators = {
            "question": "❓",
            "citation_request": "📚",
            "claim": "⚖️",
            "objection": "🚫",
            "general": "💭"
        }
        
        indicator = type_indicators.get(speech_type, "💬")
        
        formatted = f"""
{indicator} [bold]Legal Response[/bold] ([dim]{speech_type}, {processing_time:.2f}s[/dim])
[green]{response_text}[/green]
"""
        return formatted.strip()
    
    def get_statistics(self) -> Dict:
        """Get response generation statistics."""
        avg_response_time = (
            self.total_response_time / self.total_requests
            if self.total_requests > 0 else 0
        )
        
        success_rate = (
            (self.total_requests - self.failed_requests) / self.total_requests * 100
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "success_rate_percent": success_rate,
            "total_response_time": self.total_response_time,
            "average_response_time": avg_response_time
        }
    
    def print_statistics(self) -> None:
        """Print response generation statistics to console."""
        stats = self.get_statistics()
        
        console.print("\n[bold]Response Generation Statistics:[/bold]")
        console.print(f"  Total requests: {stats['total_requests']}")
        console.print(f"  Success rate: {stats['success_rate_percent']:.1f}%")
        console.print(f"  Avg response time: {stats['average_response_time']:.2f}s")
        console.print(f"  Total generation time: {stats['total_response_time']:.1f}s")


async def test_responder():
    """Test function for the legal responder."""
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]OPENAI_API_KEY environment variable not set[/red]")
        return
    
    responder = LegalResponder(api_key)
    
    # Test cases with different speech types
    test_cases = [
        {
            "speech": "What is the statute of limitations for contract disputes?",
            "context": "[Document: Contract Law Guide | Score: 0.95]\nThe statute of limitations for breach of contract claims is generally 4 years from the date of breach under UCC Section 2-725.",
            "patterns": {"is_question": True, "is_citation_request": False, "is_claim": False, "is_objection": False}
        },
        {
            "speech": "I object to this line of questioning as irrelevant.",
            "context": "No relevant documents found.",
            "patterns": {"is_question": False, "is_citation_request": False, "is_claim": False, "is_objection": True}
        },
        {
            "speech": "Can you cite the case that established the reasonable person standard?",
            "context": "[Document: Tort Law Cases | Score: 0.89]\nThe reasonable person standard was established in Vaughan v. Menlove (1837), defining the objective standard of care in negligence cases.",
            "patterns": {"is_question": True, "is_citation_request": True, "is_claim": False, "is_objection": False}
        },
        {
            "speech": "The defendant clearly violated their fiduciary duty.",
            "context": "[Document: Fiduciary Law | Score: 0.92]\nFiduciary duty requires loyalty, care, and good faith. Breach occurs when the fiduciary acts in self-interest or fails to act in the beneficiary's best interest.",
            "patterns": {"is_question": False, "is_citation_request": False, "is_claim": True, "is_objection": False}
        }
    ]
    
    console.print("[bold]Testing legal response generation:[/bold]")
    
    for i, test_case in enumerate(test_cases, 1):
        console.print(f"\n[cyan]Test Case {i}:[/cyan]")
        console.print(f"[dim]Speech:[/dim] {test_case['speech']}")
        console.print(f"[dim]Patterns:[/dim] {test_case['patterns']}")
        
        try:
            response = await responder.generate_response(
                test_case["speech"],
                test_case["context"],
                test_case["patterns"]
            )
            
            formatted = responder.format_response_for_display(response, test_case["speech"])
            console.print(formatted)
            
        except Exception as e:
            console.print(f"[red]Test failed: {e}[/red]")
    
    # Test quick response
    console.print("\n[cyan]Testing quick response:[/cyan]")
    quick_response = await responder.generate_quick_response("Objection, hearsay!")
    console.print(f"Quick: {quick_response['response_text']}")
    
    responder.print_statistics()


if __name__ == "__main__":
    asyncio.run(test_responder()) 