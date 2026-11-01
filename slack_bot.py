"""Slack bot: answers policy questions in a channel and names the policy (Method 3, vector index).

Runs locally in Socket Mode, so no public server is needed. Setup:
1. At api.slack.com/apps, create an app "From a manifest" and paste slack_manifest.yaml.
2. Basic Information -> App-Level Tokens: create a token with connections:write (starts xapp-).
3. Install the app to the workspace and copy the Bot User OAuth Token (starts xoxb-).
4. Put both in a .env file outside the repository:
       SLACK_APP_TOKEN=xapp-...
       SLACK_BOT_TOKEN=xoxb-...
5. Run  .venv/bin/python slack_bot.py  and add the bot to a channel (channel -> Integrations).
6. Mention it:  @Policy Assistant How many vacation days do I get?
"""
import os
import re
from pathlib import Path

from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

import method3_vector
from policies import load_policies

TEXT = {p["title"]: p["policy_text"] for p in load_policies()}


def env(name):
    if os.environ.get(name):
        return os.environ[name]
    for folder in Path(__file__).resolve().parents:
        f = folder / ".env"
        if f.is_file():
            for line in f.read_text().splitlines():
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip("'\"")
    raise RuntimeError(f"{name} not found")


app = App(token=env("SLACK_BOT_TOKEN"))


def reply_text(question):
    r = method3_vector.answer(question)
    lines = [r["answer"]]
    if r["policy"] in TEXT:
        lines.append(f"*Relevant policy:* {r['policy']}: \"{TEXT[r['policy']]}\"")
    else:
        lines.append("*Relevant policy:* none in the policy database")
    lines.append(f"_Answered only from the company policy database ({r['seconds']:.1f} s, "
                 f"{r['tokens_in'] + r['tokens_out']} tokens). Check the policy before acting on it._")
    return "\n".join(lines)


@app.event("app_mention")
def on_mention(event, say):
    question = re.sub(r"<@[^>]+>", "", event.get("text", "")).strip()
    thread = event.get("thread_ts") or event["ts"]
    if not question:
        say(text="Ask me a question about company policy, e.g. *How many vacation days do I get?*", thread_ts=thread)
        return
    try:
        say(text=reply_text(question), thread_ts=thread)
    except Exception as e:
        say(text=f"Sorry, I could not answer right now ({str(e)[:100]}).", thread_ts=thread)


if __name__ == "__main__":
    print("Policy bot running. Mention it in a channel it has been added to. Ctrl+C stops it.")
    SocketModeHandler(app, env("SLACK_APP_TOKEN")).start()
