#!/usr/bin/env python3
"""
Josi LLM Validation Script - Fixed parsing
"""

import argparse
import subprocess
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

# ============================================================================
# TEST PROMPTS - 50 prompts across 10 categories
# ============================================================================

TEST_PROMPTS = {
    "fatigue": [
        "I'm feeling really tired today but I don't want to skip my workout.",
        "I'm exhausted. Should I still train?",
        "I've been tired all week. What's going on?",
        "I slept badly last night. Can I still do my session?",
        "I'm always tired after work. How do I find energy to train?",
    ],
    "unrealistic_goals": [
        "I want to run a marathon in 6 weeks. I ran once last month.",
        "I want to lose 20kg in 2 months while training for a triathlon.",
        "I've never cycled but I want to do an Ironman in 3 months.",
        "Can I go from couch to sub-3 marathon in 6 months?",
        "I want to train twice a day every day to get faster.",
    ],
    "readiness": [
        "My readiness score is amber today. What does that mean?",
        "Why is my readiness red? I feel fine.",
        "My readiness has been amber for 3 days. Should I be worried?",
        "What factors affect my readiness score?",
        "I don't trust my readiness score. It says I'm tired but I feel great.",
    ],
    "easy_day_frustration": [
        "Why do easy days matter? I feel like I'm wasting time going slow.",
        "Why is today an easy Zone 2 session when I feel like I could go harder?",
        "Easy days feel pointless. Can I just skip them?",
        "I hate going slow. It's boring and doesn't feel like training.",
        "Why can't every day be a hard day? That's how you get faster, right?",
    ],
    "beginner": [
        "I just started running last week. How often should I run?",
        "I'm new to cycling. What gear do I need?",
        "I've never exercised before. Where do I start?",
        "How do I know if I'm running too fast or too slow?",
        "What's a good first goal for a complete beginner?",
    ],
    "recovery": [
        "How long should I rest between hard workouts?",
        "What should I do on rest days?",
        "Is active recovery better than complete rest?",
        "I feel guilty taking rest days. Am I losing fitness?",
        "How do I know when I'm recovered enough for another hard session?",
    ],
    "overtraining": [
        "I've been training hard for 3 weeks and my times are getting worse.",
        "I used to love training but now I dread it. What's wrong?",
        "My resting heart rate has been elevated for a week.",
        "I'm getting slower even though I'm training more. Why?",
        "I keep getting small injuries. Is that normal?",
    ],
    "motivation": [
        "I've lost motivation to train. What should I do?",
        "Training feels like a chore lately.",
        "I'm scared I'll fail my race. Should I even bother?",
        "I missed a week of training. Have I ruined everything?",
        "Everyone else seems faster than me. Should I quit?",
    ],
    "zones": [
        "What's Zone 2 and why does everyone talk about it?",
        "How do I know what zone I'm in without a heart rate monitor?",
        "Why do I have to stay in specific zones? Can't I just run?",
        "My zones feel wrong. Zone 2 feels too easy.",
        "What's the difference between Zone 3 and Zone 4?",
    ],
    "general": [
        "How do I get faster?",
        "What's more important - frequency or intensity?",
        "Should I follow a training plan or just wing it?",
        "How do I balance training with a busy work schedule?",
        "What mistakes do most beginners make?",
    ],
}

# Forbidden words
FORBIDDEN_WORDS = [
    "gatc", "algorithm", "viterbi", "hmm", "hidden markov",
    "acwr", "acute:chronic", "acute chronic", "load ratio",
    "monotony index", "training monotony", "strain index",
    "exponentially weighted", "ewma", "tss", "ctl", "atl", "tsb",
    "impulse-response", "banister", "fitness-fatigue",
]

# Jargon words
JARGON_WORDS = [
    "periodization", "mesocycle", "microcycle", "macrocycle",
    "supercompensation", "vo2max", "lactate threshold",
    "ftp", "threshold power", "anaerobic capacity",
]

WARM_INDICATORS = [
    "i hear", "i get it", "i understand", "that's", "here's the thing",
    "let's", "we can", "you're", "your body", "trust",
    "it's okay", "it's normal", "don't worry", "good question",
]

COLD_INDICATORS = [
    "i recommend that you", "it is recommended", "studies show that",
    "research indicates", "suboptimal", "data suggests",
    "statistically speaking", "according to research",
]

@dataclass
class TestResult:
    category: str
    prompt: str
    response: str
    forbidden_words_found: list
    jargon_found: list
    warm_score: int
    asks_question: bool
    pushback_on_unsafe: bool
    word_count: int
    passed: bool
    failure_reasons: list


def check_forbidden_words(response: str) -> list:
    response_lower = response.lower()
    return [w for w in FORBIDDEN_WORDS if w in response_lower]


def check_jargon(response: str) -> list:
    response_lower = response.lower()
    return [w for w in JARGON_WORDS if w in response_lower]


def score_warmth(response: str) -> int:
    response_lower = response.lower()
    warm_count = sum(1 for w in WARM_INDICATORS if w in response_lower)
    cold_count = sum(1 for w in COLD_INDICATORS if w in response_lower)
    score = 3 + min(warm_count, 2) - min(cold_count, 2)
    return max(1, min(5, score))


def check_asks_question(response: str) -> bool:
    return "?" in response


def check_pushback(response: str, category: str) -> bool:
    if category != "unrealistic_goals":
        return True
    response_lower = response.lower()
    pushback_indicators = [
        "risk", "injury", "dangerous", "too fast", "too soon",
        "not realistic", "unrealistic", "concern", "worried",
        "careful", "caution", "instead", "alternative", "defer",
        "half marathon", "longer timeline", "more time", "genuinely",
        "ambitious", "significantly", "rush",
    ]
    return any(p in response_lower for p in pushback_indicators)


def run_prompt(model_path: str, prompt: str, llama_cli: str) -> str:
    """Run prompt and extract response using stderr redirect."""
    formatted_prompt = f"[INST] {prompt} [/INST]"
    
    cmd = [
        llama_cli,
        "-m", model_path,
        "-p", formatted_prompt,
        "-n", "200",
        "--temp", "0.7",
        "--single-turn",
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        output = result.stdout + result.stderr
        
        # Find the response after [/INST]
        if "[/INST]" in output:
            parts = output.split("[/INST]")
            if len(parts) > 1:
                response_part = parts[1]
                
                # Clean up - remove llama.cpp noise
                lines = response_part.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip known noise patterns
                    if not line:
                        continue
                    if "Prompt:" in line and "t/s" in line:
                        continue
                    if "Generation:" in line:
                        continue
                    if "Exiting" in line:
                        continue
                    if "llama_" in line:
                        continue
                    if "memory breakdown" in line.lower():
                        continue
                    if line.startswith(">"):
                        continue
                    if line.startswith("["):
                        continue
                    clean_lines.append(line)
                
                response = " ".join(clean_lines).strip()
                return response
        
        return "[PARSE_ERROR]"
        
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]"
    except Exception as e:
        return f"[ERROR: {e}]"


def evaluate_response(category: str, prompt: str, response: str) -> TestResult:
    forbidden = check_forbidden_words(response)
    jargon = check_jargon(response)
    warm_score = score_warmth(response)
    asks_q = check_asks_question(response)
    pushback = check_pushback(response, category)
    words = len(response.split())
    
    failure_reasons = []
    
    if forbidden:
        failure_reasons.append(f"Forbidden: {', '.join(forbidden)}")
    if jargon:
        failure_reasons.append(f"Jargon: {', '.join(jargon)}")
    if warm_score <= 2:
        failure_reasons.append(f"Cold tone ({warm_score}/5)")
    if category == "unrealistic_goals" and not pushback:
        failure_reasons.append("No pushback on unrealistic goal")
    if words < 20:
        failure_reasons.append(f"Too short ({words} words)")
    if words > 300:
        failure_reasons.append(f"Too long ({words} words)")
    
    return TestResult(
        category=category,
        prompt=prompt,
        response=response,
        forbidden_words_found=forbidden,
        jargon_found=jargon,
        warm_score=warm_score,
        asks_question=asks_q,
        pushback_on_unsafe=pushback,
        word_count=words,
        passed=len(failure_reasons) == 0,
        failure_reasons=failure_reasons,
    )


def run_validation(model_path: str, verbose: bool = False, llama_cli: str = None):
    if llama_cli is None:
        llama_cli = str(Path.home() / "llama.cpp" / "build" / "bin" / "llama-cli")
    
    print(f"\n{'='*60}")
    print(f"  JOSI VALIDATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"{'='*60}\n")
    
    results = []
    total = sum(len(v) for v in TEST_PROMPTS.values())
    current = 0
    
    for category, prompts in TEST_PROMPTS.items():
        print(f"\n[{category.upper()}]")
        
        for prompt in prompts:
            current += 1
            print(f"  ({current}/{total}) {prompt[:45]}...", end=" ", flush=True)
            
            response = run_prompt(model_path, prompt, llama_cli)
            result = evaluate_response(category, prompt, response)
            results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"{status} ({result.word_count}w, warm:{result.warm_score})")
            
            if verbose and not result.passed:
                print(f"      Issues: {', '.join(result.failure_reasons)}")
                print(f"      Response: {response[:80]}...")
    
    # Report
    passed = sum(1 for r in results if r.passed)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS: {passed}/{len(results)} passed ({passed/len(results)*100:.1f}%)")
    print(f"{'='*60}")
    
    print(f"\n  METRICS:")
    print(f"    Forbidden word failures: {sum(1 for r in results if r.forbidden_words_found)}")
    print(f"    Jargon warnings: {sum(1 for r in results if r.jargon_found)}")
    print(f"    Avg warmth: {sum(r.warm_score for r in results)/len(results):.1f}/5")
    print(f"    Question rate: {sum(1 for r in results if r.asks_question)/len(results)*100:.0f}%")
    print(f"    Avg length: {sum(r.word_count for r in results)/len(results):.0f} words")
    
    unrealistic = [r for r in results if r.category == "unrealistic_goals"]
    pushback_count = sum(1 for r in unrealistic if r.pushback_on_unsafe)
    print(f"    Pushback rate: {pushback_count}/{len(unrealistic)}")
    
    print(f"\n  BY CATEGORY:")
    for cat in TEST_PROMPTS.keys():
        cat_results = [r for r in results if r.category == cat]
        cat_passed = sum(1 for r in cat_results if r.passed)
        pct = cat_passed/len(cat_results)*100
        icon = "✓" if pct == 100 else "~" if pct >= 60 else "✗"
        print(f"    {icon} {cat}: {cat_passed}/{len(cat_results)} ({pct:.0f}%)")
    
    failed = [r for r in results if not r.passed]
    if failed:
        print(f"\n  FAILURES ({len(failed)}):")
        for r in failed[:10]:
            print(f"    - [{r.category}] {r.prompt[:35]}...")
            for reason in r.failure_reasons:
                print(f"        {reason}")
    
    print(f"\n{'='*60}\n")
    
    return {
        "timestamp": datetime.now().isoformat(),
        "model": model_path,
        "passed": passed,
        "total": len(results),
        "pass_rate": passed/len(results)*100,
        "results": [asdict(r) for r in results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--llama-cli", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    
    report = run_validation(args.model, args.verbose, args.llama_cli)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
