"""
Reads agent.db and prints metrics you can cite in a resume/interview:
  - % of emails auto-handled vs escalated to human
  - human override rate (how often the agent's suggestion was rejected/edited)
  - average confidence per intent class
  - action distribution

Run after using the agent for a while:
    python eval.py
"""

import sqlite3
from collections import Counter, defaultdict

from tools import DB_PATH


def main():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """SELECT intent, confidence, suggested_action, final_action, human_overrode
           FROM logs"""
    ).fetchall()
    conn.close()

    if not rows:
        print("No logged decisions yet. Run `python run.py` first.")
        return

    total = len(rows)
    overrides = sum(r[4] for r in rows)
    escalated = sum(1 for r in rows if r[1] < 0.7 or r[0] == "urgent")
    auto_handled = total - escalated

    intent_confidences = defaultdict(list)
    action_counts = Counter()
    for intent, confidence, suggested_action, final_action, human_overrode in rows:
        intent_confidences[intent].append(confidence)
        action_counts[final_action] += 1

    print("=" * 60)
    print(f"Total emails processed:       {total}")
    print(f"Auto-handled (no human):      {auto_handled} ({100*auto_handled/total:.1f}%)")
    print(f"Escalated to human:           {escalated} ({100*escalated/total:.1f}%)")
    print(f"Human override rate:          {overrides} ({100*overrides/total:.1f}%)")
    print("-" * 60)
    print("Average confidence by intent:")
    for intent, confidences in intent_confidences.items():
        avg = sum(confidences) / len(confidences)
        print(f"  {intent:15s} avg={avg:.2f}  n={len(confidences)}")
    print("-" * 60)
    print("Final action distribution:")
    for action, count in action_counts.most_common():
        print(f"  {action:15s} {count} ({100*count/total:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
