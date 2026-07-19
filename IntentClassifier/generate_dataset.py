import random
import pandas as pd

random.seed(42)

# ==========================================================
# 1. Entity pools (expanded per suggestions doc)
# ==========================================================

apps = [
    # browsers
    "Chrome", "Google Chrome", "Chrome Canary", "Chrome Beta", "Firefox", "Edge", "Brave", "Opera",
    # editors / IDEs
    "VS Code", "Visual Studio Code", "Visual Studio", "PyCharm", "IntelliJ", "Android Studio", "CLion", "WebStorm", "Sublime Text", "Atom",
    # comms
    "Discord", "Slack", "Teams", "Zoom", "Skype", "WhatsApp", "Telegram",
    # gaming
    "Steam", "Epic Games", "Battle.net", "Minecraft Launcher", "Origin", "GOG Galaxy",
    # media
    "Spotify", "Apple Music", "VLC", "Windows Media Player", "YouTube Music",
    # office
    "Word", "Excel", "PowerPoint", "OneNote", "Outlook", "Google Docs", "Google Sheets",
    # creative
    "Photoshop", "Illustrator", "Premiere Pro", "After Effects", "Lightroom", "Figma", "Canva",
    # system tools
    "File Explorer", "Task Manager", "Control Panel", "Settings", "Terminal", "CMD", "PowerShell", "Git Bash",
    # misc utility
    "Notepad", "Calculator", "Paint", "Snipping Tool",
]

projects = [
    "Aquina", "HackArena", "ParcelVision", "RideWise", "SummAize", "Portfolio",
    "EcoWave", "AlphaWave", "CareBridgeAI", "NeoScore", "ContractPulse", "MoodQuest",
    "FabrAIc", "FinSkin", "MarketShield",
    "DBMS Project", "OS Lab", "Compiler Project", "React App", "Python Project", "ML Project",
    "Semester4 Project", "IEEE Paper", "Internship Project", "Resume Website", "Portfolio Site",
    "Calculator App", "Weather App", "ChatBot", "Flask API", "Todo App", "Blog Site",
    "E-commerce Site", "Chat App", "Quiz App", "Notes App", "Expense Tracker", "Recipe App",
    "Movie Recommender", "Music Player App", "Budget Tracker", "Habit Tracker", "Portfolio v2",
]

# base names combined with variants/extensions to reach 300+ without hardcoding every single one
_file_basenames = [
    "resume", "resume_final", "resume_latest", "notes", "dbms_notes", "os_lab", "todo",
    "meeting_notes", "invoice", "invoice_2026", "requirements", "package", "Dockerfile",
    "README", "LICENSE", "main", "app", "utils", "index", "style", "script", "report",
    "ppt_final", "budget", "config", "test_cases", "assignment1", "assignment2", "lab_manual",
    "project_proposal", "presentation_draft", "cover_letter", "syllabus", "timetable",
    "todo_list", "shopping_list", "notes_final", "summary", "draft", "final_report",
]
_file_extensions = [".pdf", ".docx", ".txt", ".py", ".json", ".md", ".pptx", ".xlsx", ".html", ".css", ".js", ""]
files = sorted({f"{b}{e}" for b in _file_basenames for e in _file_extensions})  # 300+ combinations

folders = [
    "Projects", "Downloads", "Assignments", "Images", "AI", "Hackathon", "College", "Desktop",
    "Documents", "Music", "Videos", "Semester4", "DBMS", "OS", "React", "Python", "Java",
    "IEEE", "Internship", "Resume", "Portfolio", "Pictures", "Vacation", "Family", "Games", "Photos",
    "Work", "School", "Backup", "Archive", "Screenshots", "Notes",
]

commands = [
    "git pull", "git status", "git push", "git add .", "git commit", "git clone",
    "git checkout main", "git merge", "git fetch",
    "npm install", "npm run dev", "npm start", "npm test",
    "python app.py", "python main.py", "flask run",
    "cargo build", "cargo run",
    "pip install", "pip freeze", "conda activate",
    "docker compose up", "docker build", "code .", "explorer .",
]

url_targets = ["google.com", "github.com", "youtube.com", "gmail.com", "chatgpt.com",
               "stackoverflow.com", "leetcode.com", "linkedin.com"]
web_search_queries = ["best laptops 2026", "how to center a div", "python vs javascript",
                      "weather bengaluru", "cheap flights to goa", "next.js tutorial"]

# casual/abbreviated forms + speech-recognition-style errors
abbreviations = {
    "VS Code": ["vscode", "vs code", "code", "vsc", "vs"],
    "Visual Studio Code": ["vscode"],
    "Photoshop": ["ps", "photo editor", "adobe ps"],
    "PowerPoint": ["ppt", "powerpoint"],
    "File Explorer": ["explorer", "file explorer"],
    "Terminal": ["term", "terminal", "cmd"],
    "Android Studio": ["android studio", "studio"],
    "Google Chrome": ["chrome", "gchrome"],
}

typos = {
    "Chrome": ["Chorme", "Chrom", "Chrme"],
    "Spotify": ["Spotfy", "Spotifiy"],
    "Discord": ["Discrod", "Disocrd"],
    "Calculator": ["Calculater", "Calc"],
    "PowerPoint": ["Power Pont", "Powerpoint"],
    "VS Code": ["VSCode", "VS-Code", "VSCODE", "vscode"],
}

# phonetic/speech-recognition-style misheard forms (voice input use case)
speech_errors = {
    "Chrome": ["crowm", "krom"],
    "VS Code": ["vs codd", "vees code"],
    "Spotify": ["spottyfy", "spot a fy"],
    "File Explorer": ["file explorerer"],
    "Terminal": ["terminull"],
}

# ==========================================================
# 2. Templates per intent — original six kept, 29 new ones added
# ==========================================================

templates = {
    "open_app": [
        "open {x}", "launch {x}", "start {x}", "run {x}", "boot {x}", "fire up {x}",
        "bring up {x}", "open up {x}", "I need {x}", "I'd like to open {x}",
        "could you please open {x}", "can you launch {x} for me", "hey, pull up {x}",
        "get {x} open", "I want to use {x}", "switch to {x}", "let's open {x}",
        "spin up {x}", "get {x} running", "load {x}",
    ],
    "close_app": [
        "close {x}", "quit {x}", "exit {x}", "terminate {x}", "kill {x}", "stop {x}",
        "shut down {x}", "close down {x}", "can you close {x}", "I need {x} shut down now",
        "please quit {x}", "get rid of {x}", "shut {x} off", "end {x}",
    ],
    "search_file": [
        "find {x}", "locate {x}", "search for {x}", "where is {x}", "look for {x}",
        "open file {x}", "find file {x}", "can you find {x} for me",
        "do we have a file called {x}", "pull up {x}", "I'm looking for {x}",
    ],
    "create_folder": [
        "create folder {x}", "make folder {x}", "new folder {x}", "create directory {x}",
        "make a folder called {x}", "can you create a folder named {x}",
        "I need a new folder called {x}", "set up a folder for {x}",
    ],
    "open_project": [
        "open project {x}", "load project {x}", "launch project {x}", "start project {x}",
        "open my {x} project", "can you open the {x} project", "pull up {x} project",
        "let's work on {x}", "switch to my {x} project",
    ],
    "run_command": [
        "run {x}", "execute {x}", "run command {x}", "execute command {x}",
        "start command {x}", "can you run {x}", "please execute {x}",
    ],

    # --- new file/folder operations ---
    "delete_file": ["delete {x}", "remove {x}", "get rid of the file {x}", "trash {x}", "can you delete {x}"],
    "delete_folder": ["delete folder {x}", "remove folder {x}", "trash the {x} folder", "get rid of the {x} folder"],
    "rename_file": ["rename {x} to newname", "rename the file {x}", "can you rename {x}"],
    "rename_folder": ["rename folder {x}", "rename the {x} folder"],
    "copy_file": ["copy {x}", "duplicate {x}", "make a copy of {x}"],
    "move_file": ["move {x} to another folder", "move {x} to Desktop", "can you move {x}"],

    # --- web ---
    "open_url": ["open {x}", "go to {x}", "take me to {x}", "open the site {x}", "visit {x}"],
    "search_web": ["search for {x}", "google {x}", "look up {x}", "search the web for {x}"],

    # --- settings/system ---
    "open_settings": ["open settings", "take me to settings", "I need to change a setting", "open system settings"],
    "shutdown_pc": ["shutdown", "shut down the pc", "turn off my computer", "power off", "shutdown now"],
    "restart_pc": ["restart", "reboot the pc", "restart my computer", "reboot now"],
    "sleep_pc": ["put the pc to sleep", "sleep mode", "go to sleep", "sleep now"],
    "lock_pc": ["lock my pc", "lock the screen", "lock it", "lock now"],
    "logout": ["log me out", "sign out", "logout now", "log out of this account"],

    # --- media/volume ---
    "volume_up": ["volume up", "turn the volume up", "increase volume", "make it louder"],
    "volume_down": ["volume down", "turn the volume down", "lower the volume", "make it quieter"],
    "mute": ["mute", "mute the sound", "turn off sound", "silence it"],
    "unmute": ["unmute", "turn sound back on", "unmute it"],
    "brightness_up": ["brightness up", "increase brightness", "make the screen brighter"],
    "brightness_down": ["brightness down", "lower the brightness", "dim the screen"],
    "play_music": ["play music", "play something", "start playing music", "play a song"],
    "pause_music": ["pause music", "pause the song", "pause it"],
    "resume_music": ["resume music", "continue playing", "unpause it"],
    "next_song": ["next song", "skip this song", "play the next track"],
    "previous_song": ["previous song", "go back a track", "play the last song"],

    # --- screen/clipboard ---
    "take_screenshot": ["take a screenshot", "capture the screen", "screenshot this"],
    "clipboard_history": ["show clipboard history", "what did I copy earlier", "open clipboard manager"],
    "empty_recycle_bin": ["empty the recycle bin", "clear trash", "empty trash"],

    # --- unknown / gibberish ---
    "unknown": ["asdfgh", "12345", "......", "banana potato", "random text", "xyzxyz",
                "?????", "abcdefg", "hmmmm", "what", "idk", "lorem ipsum", "qwerty"],
}

mapping = {
    "open_app": apps, "close_app": apps,
    "search_file": files, "delete_file": files, "rename_file": files, "copy_file": files, "move_file": files,
    "create_folder": folders, "delete_folder": folders, "rename_folder": folders,
    "open_project": projects,
    "run_command": commands,
    "open_url": url_targets,
    "search_web": web_search_queries,
}
# intents with no slot values (fixed action, sentence has no {x})
NO_SLOT_INTENTS = [
    "open_settings", "shutdown_pc", "restart_pc", "sleep_pc", "lock_pc", "logout",
    "volume_up", "volume_down", "mute", "unmute", "brightness_up", "brightness_down",
    "play_music", "pause_music", "resume_music", "next_song", "previous_song",
    "take_screenshot", "clipboard_history", "empty_recycle_bin", "unknown",
]

# ==========================================================
# 3. general_chat — includes hard negatives (expanded)
# ==========================================================

chat_plain = [
    "hello", "hi", "hey there", "good morning", "good evening", "how are you",
    "who are you", "tell me a joke", "thank you", "thanks a lot", "what can you do",
    "what's the weather today", "how do I fix this bug", "explain recursion to me",
    "what's on my calendar today", "remind me about my meeting",
]

chat_hard_negatives_templates = [
    "I really like using {x}", "{x} is my favorite app", "have you used {x} before",
    "what do you think of {x}", "I used to use {x} a lot", "{x} crashed on me yesterday",
    "is {x} better than its competitors", "I'm learning {x} right now",
    "my project {x} is going well", "I'm proud of how {x} turned out",
    "tell me about the {x} project", "what tech stack did I use for {x}",
    "{x} uses too much RAM", "{x} is amazing", "{x} isn't opening for some reason",
    "I like {x}", "{x} is confusing", "{x} has a memory leak",
    "what is {x}", "how does {x} work", "can you explain {x}",
]

# ==========================================================
# 4. Ambiguous / disambiguation examples
# ==========================================================

ambiguous_pairs = []
for p in projects:
    ambiguous_pairs.append((f"open {p}", "open_project", p))
    ambiguous_pairs.append((f"open my {p} project", "open_project", p))
for a in apps:
    ambiguous_pairs.append((f"open {a}", "open_app", a))
    ambiguous_pairs.append((f"open {a} please", "open_app", a))

# ==========================================================
# 5. Multi-intent commands (compound sentences)
# connector joins two single-action phrases; slot stores both
# entities found, semicolon-separated, so downstream parsing
# knows there are multiple targets to act on in sequence.
# ==========================================================

connectors = [" and ", " then ", " and then ", ", ", " before ", " after ", " followed by "]

multi_intent_pairs = [
    (("open {a}", apps), ("open {b}", apps)),
    (("close {a}", apps), ("open {b}", apps)),
    (("launch {a}", apps), ("launch {b}", apps)),
    (("open project {a}", projects), ("run {b}", commands)),
    (("open {a}", apps), ("run {b}", commands)),
    (("find {a}", files), ("open it", None)),
    (("create folder {a}", folders), ("open it", None)),
    (("run {a}", commands), ("run {b}", commands)),
    (("shutdown", None), ("close {b}", apps)),
]

# ==========================================================
# 6. Noise / conversational-variation / context wrapper
# ==========================================================

fillers = ["", "please ", "hey ", "can you ", "could you ", "would you ", "quickly ",
           "kindly ", "so, ", "um, ", "uh, ", "actually, ", "okay, ", "ok, ", "yo ",
           "bro ", "pls ", "plz "]
tails = ["", "", "", " please", " now", " asap", " for me", " if possible",
         " when you can", " thanks", " ty", " right now", " real quick", " lol"]

reasons = [
    "because I need Gmail", "for today's assignment", "it's lagging",
    "I forgot where it is", "for the DBMS lab", "so I can browse",
    "for today's work", "before the meeting", "I need it urgently",
]


def add_noise(sentence, n=4, allow_context=True):
    out = []
    used_filler_word = any(w in sentence.lower() for w in
                            ["please", "can you", "could you", "would you"])
    for _ in range(n):
        filler = "" if used_filler_word else random.choice(fillers)
        tail = random.choice(tails)
        t = f"{filler}{sentence}{tail}".strip()

        # occasionally append a context/reason clause
        if allow_context and random.random() < 0.12:
            t = f"{t} {random.choice(reasons)}"

        # punctuation variety
        r = random.random()
        if r < 0.20:
            t += "."
        elif r < 0.30:
            t += "!"
        elif r < 0.36:
            t += "!!"
        elif r < 0.40:
            t += "..."
        elif r < 0.44:
            t += "??"

        # casing variety
        r2 = random.random()
        if r2 < 0.30:
            t = t.lower()
        elif r2 < 0.35:
            t = t.upper()
        elif r2 < 0.38:
            # random mixed case, e.g. "cHrOmE" style typing accident
            t = "".join(c.upper() if random.random() < 0.5 else c.lower() for c in t)

        out.append(t.strip())
    return out


# ==========================================================
# 7. Build dataset — fixed sampling budget per intent
# (prevents combinatorial blow-up now that pools are 100-300+ items)
# ==========================================================

RAW_PER_SLOT_INTENT = 450     # target raw rows per intent that has slot values
RAW_PER_NOSLOT_INTENT = 150   # smaller pools, need fewer raw rows
NOISE_PER_SAMPLE = 3

rows = []

for intent, temp_list in templates.items():
    if intent in NO_SLOT_INTENTS:
        target = RAW_PER_NOSLOT_INTENT
        for _ in range(target // NOISE_PER_SAMPLE):
            temp = random.choice(temp_list)
            for v in add_noise(temp, n=NOISE_PER_SAMPLE):
                rows.append({"text": v, "intent": intent, "slot": ""})
        continue

    values = mapping[intent]
    target = RAW_PER_SLOT_INTENT
    for _ in range(target // NOISE_PER_SAMPLE):
        temp = random.choice(temp_list)
        value = random.choice(values)
        sentence = temp.format(x=value)
        for v in add_noise(sentence, n=NOISE_PER_SAMPLE):
            rows.append({"text": v, "intent": intent, "slot": value})

        # occasionally substitute an abbreviation / typo / speech-error variant
        r = random.random()
        if value in abbreviations and r < 0.15:
            alt = random.choice(abbreviations[value])
            alt_sentence = temp.format(x=alt)
            for v in add_noise(alt_sentence, n=1):
                rows.append({"text": v, "intent": intent, "slot": value})
        elif value in typos and r < 0.30:
            alt = random.choice(typos[value])
            alt_sentence = temp.format(x=alt)
            for v in add_noise(alt_sentence, n=1):
                rows.append({"text": v, "intent": intent, "slot": value})
        elif value in speech_errors and r < 0.45:
            alt = random.choice(speech_errors[value])
            alt_sentence = temp.format(x=alt)
            for v in add_noise(alt_sentence, n=1):
                rows.append({"text": v, "intent": intent, "slot": value})

# general chat — plain
for s in chat_plain:
    for v in add_noise(s, n=6, allow_context=False):
        rows.append({"text": v, "intent": "general_chat", "slot": ""})

# general chat — hard negatives (mentions an app/project but isn't a command)
for _ in range(600):
    temp = random.choice(chat_hard_negatives_templates)
    x = random.choice(apps + projects)
    sentence = temp.format(x=x)
    for v in add_noise(sentence, n=1, allow_context=False):
        rows.append({"text": v, "intent": "general_chat", "slot": ""})

# explicit ambiguous/disambiguation examples
for _ in range(500):
    sentence, intent, slot = random.choice(ambiguous_pairs)
    for v in add_noise(sentence, n=1):
        rows.append({"text": v, "intent": intent, "slot": slot})

# multi-intent compound commands
for _ in range(500):
    (t1, pool1), (t2, pool2) = random.choice(multi_intent_pairs)
    val_a = random.choice(pool1) if pool1 else None
    val_b = random.choice(pool2) if pool2 else None
    part1 = t1.format(a=val_a) if "{a}" in t1 else t1
    part2 = t2.format(b=val_b) if "{b}" in t2 else t2
    connector = random.choice(connectors)
    sentence = f"{part1}{connector}{part2}"
    slots = ";".join([v for v in [val_a, val_b] if v])
    for v in add_noise(sentence, n=1, allow_context=False):
        rows.append({"text": v, "intent": "multi_intent", "slot": slots})

df = pd.DataFrame(rows).drop_duplicates(subset=["text"]).reset_index(drop=True)

# ==========================================================
# 8. Balance classes (cap largest so a few intents don't dominate)
# ==========================================================

counts = df["intent"].value_counts()
min_count = counts.min()
cap = min(counts.max(), max(min_count * 3, 300))

balanced = []
for intent, group in df.groupby("intent"):
    n = min(len(group), cap)
    balanced.append(group.sample(n=n, random_state=42))

df = pd.concat(balanced).sample(frac=1, random_state=42).reset_index(drop=True)

print("Per-intent counts after balancing:")
print(df["intent"].value_counts())
print("Total samples:", len(df))
print("Total intents:", df["intent"].nunique())

# ==========================================================
# 9. Train / val / test split
# ==========================================================

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
n = len(df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

train_df.to_csv("aquina_train.csv", index=False)
val_df.to_csv("aquina_val.csv", index=False)
test_df.to_csv("aquina_test.csv", index=False)

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ==========================================================
# 10. Reality-check holdout set — unchanged filename, expanded content
# ==========================================================

reality_check = [
    {"text": "yo pull up vscode real quick", "intent": "open_app", "slot": "VS Code"},
    {"text": "can u kill spotify its being weird", "intent": "close_app", "slot": "Spotify"},
    {"text": "wheres that resume pdf i made last week", "intent": "search_file", "slot": "resume.pdf"},
    {"text": "make a new folder for the hackathon stuff", "intent": "create_folder", "slot": "Hackathon"},
    {"text": "lets get back into aquina", "intent": "open_project", "slot": "Aquina"},
    {"text": "run the dev server", "intent": "run_command", "slot": "npm run dev"},
    {"text": "aquina has been going really well lately", "intent": "general_chat", "slot": ""},
    {"text": "whats chrome doing eating all my ram", "intent": "general_chat", "slot": ""},
    {"text": "delete that old invoice pdf", "intent": "delete_file", "slot": "invoice.pdf"},
    {"text": "can you rename my report file", "intent": "rename_file", "slot": "report.docx"},
    {"text": "go to github real quick", "intent": "open_url", "slot": "github.com"},
    {"text": "google how to center a div", "intent": "search_web", "slot": "how to center a div"},
    {"text": "put this thing to sleep", "intent": "sleep_pc", "slot": ""},
    {"text": "turn it up a bit", "intent": "volume_up", "slot": ""},
    {"text": "skip this song", "intent": "next_song", "slot": ""},
    {"text": "open chrome and then vscode", "intent": "multi_intent", "slot": "Chrome;VS Code"},
    {"text": "asdkjfh", "intent": "unknown", "slot": ""},
]
pd.DataFrame(reality_check).to_csv("aquina_reality_check.csv", index=False)
print(f"Reality-check holdout set: {len(reality_check)} examples (NOT used in training)")