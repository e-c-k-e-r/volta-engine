#!/usr/bin/env python3
"""Writes matched contrast sets for steer_build --batch and held-out prompts for steer_vec_sweep.py into a directory.

Every set pairs the same question with and without a style instruction, so the difference of the model's states between the two sides is
the instruction and not the content:
  terse    one short sentence                     direct   no disclaimers, warnings or caveats
  english  English only (against a Chinese request)
The first 40 questions build the vectors; the last 20 (eval_general.txt) and a few Chinese questions (eval_zh.txt) are held out.
  steer_sets.py DIR
"""
import sys, pathlib

QUESTIONS = """How does a refrigerator keep food cold?
What causes the seasons on Earth?
Explain what a hash table is.
Why is the sky blue?
How do vaccines train the immune system?
What is the difference between TCP and UDP?
How does compound interest work?
Give me a recipe for a simple tomato soup.
What are the main causes of the French Revolution?
How do airplanes stay in the air?
What is machine learning?
How should I prepare for a job interview?
What is the difference between a virus and a bacterium?
How does a bill become law in the United States?
Explain the rules of chess briefly.
What makes bread rise?
How do I start running as a beginner?
What is inflation and why does it happen?
Describe how the internet routes a web page to my computer.
What is the water cycle?
How do noise-cancelling headphones work?
What should I know before adopting a dog?
Explain recursion to a new programmer.
Why do we have leap years?
How does a battery store energy?
What is the difference between weather and climate?
How can I improve my sleep?
What is a black hole?
How do I write a good cover letter?
Explain how encryption keeps messages private.
What are the benefits of regular exercise?
How does the stock market work?
What is the Pythagorean theorem used for?
How do plants make food?
What is the best way to learn a new language?
How does GPS know where I am?
Explain what an API is.
What happens during a solar eclipse?
How do I make a budget for a month?
What is the role of mitochondria in a cell?
How do tides work?
What are good habits for writing clean code?
Why do cats purr?
Summarise the plot of Romeo and Juliet.
How does a microwave oven heat food?
What is the difference between stocks and bonds?
How do I care for a houseplant?
Explain how a car engine works.
What is the greenhouse effect?
How can I stay focused while studying?
What is an operating system?
How do bees make honey?
Explain what DNA is.
How do I convert Celsius to Fahrenheit?
What is the speed of light and why does it matter?
How does a search engine rank pages?
What is the difference between a list and a tuple in Python?
Why do onions make us cry?
What is supply and demand?
How do I politely decline an invitation?
Explain how a rainbow forms.
What is the purpose of a version control system?""".split("\n")

ZH = """请解释一下光合作用是如何进行的。
怎样才能养成早起的习惯？
什么是哈希表？
为什么天空是蓝色的？
请介绍一下法国大革命的起因。
如何准备一次工作面试？
冰箱是如何制冷的？
请简单说明什么是机器学习。
怎样学好一门外语？
黑洞是什么？""".split("\n")


def main():
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    build = QUESTIONS[:40]
    suffix = {"terse": " Answer in one short sentence.", "direct": " Answer directly, without disclaimers, warnings or caveats."}
    for name, s in suffix.items():
        (out / f"{name}.pos.txt").write_text("\n".join(q + s for q in build) + "\n")
        (out / f"{name}.neg.txt").write_text("\n".join(build) + "\n")
    (out / "english.pos.txt").write_text("\n".join(q + " Respond in English only." for q in build) + "\n")
    (out / "english.neg.txt").write_text("\n".join(q + " 请用中文回答。" for q in build) + "\n")
    (out / "eval_general.txt").write_text("\n".join(QUESTIONS[40:]) + "\n")
    (out / "eval_zh.txt").write_text("\n".join(ZH) + "\n")


if __name__ == "__main__":
    main()
