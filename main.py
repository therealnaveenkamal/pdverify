#!/usr/bin/env python3
"""
Main entry point for Verify-PD system.
"""

import argparse
import logging
import sys
import torch

from src.engine import SpeculativeEngine
from src.scheduler import Request, LaneType
from src.utils import get_cpu_config, get_default_config, get_test_config, VerifyPDConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify-PD: Disaggregated Speculative Decoding")
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use (cpu or cuda)"
    )
    
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Draft model name or path"
    )
    
    parser.add_argument(
        "--verifier-model",
        type=str,
        default=None,
        help="Verifier model name or path"
    )
    
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run in test mode with small models"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Explain quantum computing in simple terms.",
        help="Prompt to generate from"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Verify-PD: Disaggregated Serving for Speculative Decoding")
    logger.info("=" * 80)

    # Validate device availability early so users aren't surprised later
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        args.device = "cpu"
    
    # Get configuration
    if args.test_mode:
        config = get_test_config()
        logger.info("Running in TEST mode with small models")
    elif args.device == "cpu":
        config = get_cpu_config()
        logger.info("Running on CPU")
    else:
        config = get_default_config()
        logger.info("Running on CUDA")
    
    # Override models if specified
    if args.draft_model:
        config.model.draft_model_name = args.draft_model
    if args.verifier_model:
        config.model.verifier_model_name = args.verifier_model
    
    config.hardware.device = args.device
    
    logger.info(f"Draft model: {config.model.draft_model_name}")
    logger.info(f"Verifier model: {config.model.verifier_model_name}")
    logger.info(f"Initial draft length (L): {config.controller.initial_draft_length}")
    
    # Create and start engine
    try:
        with SpeculativeEngine(config) as engine:
            logger.info("\nProcessing request...")
            logger.info(f"Prompt: {args.prompt}")
            
            # Create request
            request = Request(
                request_id="demo_request",
                prompt=args.prompt,
                stage=LaneType.PREFILL
            )
            
            # Process request
            result = engine.process_request(request)
            
            # Display results
            logger.info("=" * 80)
            logger.info("RESULTS")
            logger.info("=" * 80)
            logger.info(f"Generated text:\n{result}")
            logger.info(f"\nTokens generated: {request.tokens_generated}")
            logger.info(f"Tokens accepted: {request.tokens_accepted}")
            logger.info(f"Acceptance ratio: {request.get_acceptance_ratio():.2%}")
            
            # Display statistics
            stats = engine.get_stats()
            logger.info("\n" + "=" * 80)
            logger.info("ENGINE STATISTICS")
            logger.info("=" * 80)
            logger.info(f"Overall acceptance rate: {stats['overall_acceptance_rate']:.2%}")
            logger.info(f"Total tokens generated: {stats['total_tokens_generated']}")
            logger.info(f"Total tokens accepted: {stats['total_tokens_accepted']}")
            logger.info(f"Requests completed: {stats['scheduler']['total_completed']}")
            logger.info(f"Current draft length: {stats['controller']['current_draft_length']}")
            logger.info("=" * 80)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
