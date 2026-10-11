import sys
import os
import re
import json
import random
import threading
from collections import Counter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTextEdit, QToolBar, QFontComboBox,
    QComboBox, QFileDialog, QColorDialog, QMessageBox, QDialog, QVBoxLayout,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget, QMenu, QInputDialog,
    QSlider, QListWidget, QListWidgetItem, QDockWidget, QProgressBar, QSpinBox,
    QTabWidget, QTime, QTableWidget, QTableWidgetItem, QHeaderView, QFormLayout,
    QGroupBox, QFrame
)
from PySide6.QtGui import (
    QAction, QFont, QTextCursor, QIcon, Qt, QKeySequence, QPixmap, QPainter, QColor,
    QPen, QTextListFormat, QTextCharFormat, QSyntaxHighlighter, QPalette,
    QTextBlockFormat, QDesktopServices, QTextTableFormat, QTextFormat,
    QTextDocument, QPageSize, QMarginsF, QPageLayout
)
from PySide6.QtCore import QSize, QSettings, QEvent, QRectF, QUrl, QThread, Signal, QTimer
from PySide6.QtPrintSupport import (
    QPrinter, QPrintDialog, QPrintPreviewDialog, QPageSetupDialog
)
from PySide6.QtTextToSpeech import QTextToSpeech

# --- Algorithm Libraries ---
try:
    from symspellpy import SymSpell, Verbosity
    SYMSPELL_AVAILABLE = True
except ImportError:
    SYMSPELL_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    import jellyfish
    JELLYFISH_AVAILABLE = True
except ImportError:
    JELLYFISH_AVAILABLE = False

try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False

# Load spaCy model safely
NLP = None
if SPACY_AVAILABLE:
    try:
        NLP = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        os.system("python -m spacy download en_core_web_sm")
        NLP = spacy.load("en_core_web_sm")

# --- Name Generator Data ---
FANTASY_NAMES = ["Aldwin", "Brenna", "Caelum", "Daenerys", "Eldrin", "Faye", "Gael", "Helena", "Isolde", "Jorah", "Kaelen", "Lyra", "Mordecai", "Nyssa", "Orion", "Perrin", "Quinn", "Rhaenyra", "Silas", "Talia", "Uther", "Vael", "Wren", "Xander", "Yvaine", "Zephyr"]
MODERN_NAMES = ["James", "Mary", "Robert", "Patricia", "Michael", "Jennifer", "William", "Linda", "David", "Elizabeth", "Richard", "Barbara", "Joseph", "Susan", "Thomas", "Jessica", "Charles", "Sarah", "Christopher", "Karen"]

# --- Mini Thesaurus Data ---
THESAURUS = {
    "big": ["large", "huge", "gigantic", "enormous", "massive"],
    "small": ["tiny", "little", "compact", "minuscule", "petite"],
    "happy": ["joyful", "cheerful", "content", "delighted", "ecstatic"],
    "sad": ["unhappy", "sorrowful", "depressed", "downcast", "miserable"],
    "fast": ["quick", "rapid", "swift", "speedy", "hasty"],
    "slow": ["sluggish", "unhurried", "leisurely", "gradual", "plodding"],
    "good": ["excellent", "fine", "superior", "positive", "favorable"],
    "bad": ["poor", "inferior", "negative", "unfavorable", "wicked"],
    "said": ["stated", "remarked", "declared", "expressed", "uttered"],
    "walk": ["stroll", "saunter", "hike", "march", "wander"],
    "run": ["sprint", "jog", "dash", "dart", "scamper"],
    "look": ["gaze", "stare", "glance", "peek", "observe"]
}

def create_text_icon(text: str, font_size: int = 14, bold: bool = True) -> QIcon:
    """Creates a simple text-based icon for toolbars."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Arial", font_size)
    if bold: font.setBold(True)
    painter.setFont(font)
    painter.setPen(QPen(Qt.GlobalColor.black))
    painter.drawText(QRectF(0, 0, 32, 32), Qt.AlignmentFlag.AlignCenter, text)
    painter.end()
    return QIcon(pixmap)

# --- Background NLP Thread for Advanced Grammar & Context ---
class NlpAnalysisThread(QThread):
    """Runs spaCy NLP analysis in the background to prevent UI freezing."""
    errors_found = Signal(list)

    def __init__(self) -> None:
        super().__init__()
        self._text = ""
        self._lock = threading.Lock()
        self._running = True

    def set_text(self, text: str) -> None:
        with self._lock:
            self._text = text

    def stop(self) -> None:
        self._running = False
        self.wait(2000)

    def run(self) -> None:
        while self._running:
            text_to_process = ""
            with self._lock:
                text_to_process = self._text
                self._text = ""

            if not NLP or not text_to_process.strip():
                self.msleep(200)
                continue

            if len(text_to_process) > 8000:
                text_to_process = text_to_process[:8000]

            try:
                doc = NLP(text_to_process)
                errors = self._analyze(doc)
                if errors and self._running:
                    self.errors_found.emit(errors)
            except Exception as e:
                print(f"NLP analysis error: {e}")
            
            self.msleep(300)

    def _analyze(self, doc) -> list:
        errors = []
        for sent in doc.sents:
            has_verb = any(t.pos_ in ["VERB", "AUX"] for t in sent)
            if not has_verb and len(sent.text.split()) > 3:
                errors.append((sent.start_char, len(sent.text.rstrip()),
                              "Grammar: Sentence Fragment (Missing verb)", "grammar"))
            for token in sent:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subj, verb = token, token.head
                    if subj.tag_ in ["NN", "NNP"] and verb.tag_ in ["VBP", "VB"]:
                        errors.append((verb.idx, len(verb.text),
                                      f"Subject-Verb Agreement: '{subj.text}' (singular) requires singular verb.",
                                      "grammar"))
                if token.dep_ in ("auxpass", "nsubjpass"):
                    verb = token.head if token.dep_ == "auxpass" else token.head.head
                    errors.append((verb.idx, len(verb.text), "Stylistic: Passive Voice (NLP)", "passive"))
                if token.text.lower() == "to" and token.i + 1 < len(doc):
                    nxt = token.nbor()
                    if nxt.pos_ in ["NOUN", "PROPN"] and nxt.text.lower() not in {"school", "work", "bed", "church"}:
                        errors.append((token.idx, len(token.text), "Contextual: Did you mean 'too'?", "grammar"))
        return errors

# --- Improved Spell, Grammar, and Style Highlighter Engine ---
class SpellGrammarHighlighter(QSyntaxHighlighter):
    """Custom syntax highlighter for spelling, grammar, and style checking."""
    def __init__(self, document) -> None:
        super().__init__(document)
        self.language = 'en_US'
        self.ignored_words = set()
        self.session_ignored = set()
        self.academic_mode = False
        self.nlp_errors = {}  

        self.sym_spell = None
        if SYMSPELL_AVAILABLE:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            try:
                import pkg_resources
                dict_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
                self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)
            except Exception as e:
                print(f"SymSpell dictionary load failed: {e}")

        self.autocorrect_map = {
            "teh": "the", "adn": "and", "recieve": "receive", "definately": "definitely",
            "seperate": "separate", "occured": "occurred", "thier": "their", "fro": "for",
            "youre": "you're", "wont": "won't", "dont": "don't", "cant": "can't"
        }

        self.load_base_words()
        self.load_custom_dictionary()

        self.spell_format = QTextCharFormat()
        self.spell_format.setUnderlineColor(QColor(231, 76, 60)) 
        self.spell_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

        self.grammar_format = QTextCharFormat()
        self.grammar_format.setUnderlineColor(QColor(52, 152, 219)) 
        self.grammar_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

        self.passive_format = QTextCharFormat()
        self.passive_format.setUnderlineColor(QColor(155, 89, 182)) 
        self.passive_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.WaveUnderline)

        self.style_format = QTextCharFormat()
        self.style_format.setUnderlineColor(QColor(39, 174, 96)) 
        self.style_format.setUnderlineStyle(QTextCharFormat.UnderlineStyle.DashDotLine)

        self.word_pattern = re.compile(r"[A-Za-z]+(?:'[a-z]+)?(?:-[A-Za-z]+)*")
        self.url_pattern = re.compile(r'\b(?:https?://|www\.)\S+\b|\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        self._build_grammar_rules()

    def _build_grammar_rules(self) -> None:
        raw_rules = [
            (r'\bi\b', "Capitalize 'I'", self.grammar_format, 0, "grammar"),
            (r'\b(alot)\b', "Should be 'a lot'", self.grammar_format, re.IGNORECASE, "grammar"),
            (r'\b(could|should|would|must|might|may)\s+of\b', "Should be 'have'", self.grammar_format, re.IGNORECASE, "grammar"),
            (r'\b(irregardless)\b', "Should be 'regardless'", self.grammar_format, re.IGNORECASE, "grammar"),
            (r'\b(loose)\s+(weight|control|mind|game|track|focus|hope|sight)\b', "Grammar: Did you mean 'lose'?", self.grammar_format, re.IGNORECASE, "grammar"),
            (r'\b(effect)\s+(you|me|him|her|them|us|the|our|my|your)\b', "Grammar: Did you mean 'affect'?", self.grammar_format, re.IGNORECASE, "grammar"),
            (r'\s+[,.!?;:]', "Space before punctuation", self.grammar_format, 0, "grammar"),
            (r'  +', "Double space", self.grammar_format, 0, "grammar"),
            (r'\b(in order to)\b', "Wordy: Consider 'to'", self.style_format, re.IGNORECASE, "style"),
            (r'\b(due to the fact that)\b', "Wordy: Consider 'because'", self.style_format, re.IGNORECASE, "style"),
            (r'\b(at this point in time)\b', "Wordy: Consider 'now'", self.style_format, re.IGNORECASE, "style"),
            (r'\b(very|really|quite|rather|somewhat|fairly|pretty)\s+\w+', "Weak intensifier", self.style_format, re.IGNORECASE, "style"),
            (r'^\s*(there)\s+(is|are|was|were)\b', "Weak opener: Consider starting with the subject", self.style_format, re.IGNORECASE, "style"),
            (r'\b(mankind)\b', "Inclusive: Consider 'humanity'", self.style_format, re.IGNORECASE, "style"),
        ]
        self.grammar_rules = []
        for pattern_str, message, fmt, flags, category in raw_rules:
            compiled = re.compile(pattern_str, flags)
            self.grammar_rules.append({'pattern': compiled, 'message': message, 'format': fmt, 'category': category})

    def set_language(self, lang_code: str) -> None:
        self.language = lang_code
        self.rehighlight()

    def set_academic_mode(self, enabled: bool) -> None:
        self.academic_mode = enabled
        if enabled:
            extra = re.compile(r"\b(can't|won't|don't|isn't|aren't|wouldn't|shouldn't|couldn't)\b", re.IGNORECASE)
            self.grammar_rules.append({
                'pattern': extra, 'message': "Academic: Avoid contractions",
                'format': self.style_format, 'category': 'style'
            })
        else:
            self.grammar_rules = [r for r in self.grammar_rules if r['message'] != "Academic: Avoid contractions"]
        self.rehighlight()

    def set_nlp_errors(self, errors: list) -> None:
        self.nlp_errors = {start: (length, msg, fmt) for start, length, msg, fmt in errors}
        self.rehighlight()

    def add_to_permanent_dictionary(self, word: str) -> None:
        word = word.lower().strip()
        if not word: return
        if word in self.ignored_words: return  # Prevent duplicates
        
        self.ignored_words.add(word)
        self.session_ignored.add(word)
        if self.sym_spell:
            self.sym_spell.create_dictionary_entry(word, 1)
        with open("custom_dictionary.txt", 'a', encoding='utf-8') as f:
            f.write(word + "\n")
        self.rehighlight()

    def is_misspelled(self, word: str) -> bool:
        lower = word.lower()
        if len(word) <= 1 or lower in self.ignored_words or lower in self.session_ignored:
            return False
        if word.isupper() and len(word) <= 5: return False
        if word[0].isupper() and not word.isupper(): return False
        if any(c.isdigit() for c in word): return False

        if self.sym_spell:
            suggestions = self.sym_spell.lookup(lower, Verbosity.CLOSEST, max_edit_distance=2)
            return len(suggestions) == 0
        else:
            return lower not in self.ignored_words  

    def get_suggestions(self, word: str) -> list:
        if not self.sym_spell: return []
        suggestions = self.sym_spell.lookup(word.lower(), Verbosity.CLOSEST, max_edit_distance=2)
        if JELLYFISH_AVAILABLE and suggestions:
            word_meta = jellyfish.metaphone(word.lower())
            suggestions.sort(key=lambda s: jellyfish.jaro_similarity(word_meta, jellyfish.metaphone(s.term)), reverse=True)
        return [s.term for s in suggestions[:5]]

    def load_base_words(self) -> None:
        tech_words = ["python", "pyside", "qt", "github", "javascript", "api", "ui", "ux", "async", "await", "smartphone", "internet", "emoji", "startup", "frontend", "backend", "openai", "chatgpt", "docker", "kubernetes"]
        literary_words = ["protagonist", "antagonist", "subplot", "foreshadowing", "archetype", "dystopia", "utopia", "memoir", "narrative", "prose", "stanza", "metaphor", "simile", "hyperbole", "onomatopoeia"]
        fantasy_words = ["dragon", "elf", "elves", "dwarves", "magic", "sorcerer", "kingdom", "sword", "castle", "griffin", "phoenix", "tavern", "spell", "wizard"]
        common_words = ["the", "and", "but", "for", "nor", "yet", "so", "in", "on", "at", "to", "with", "by", "is", "am", "are", "was", "were", "be", "being", "been", "have", "has", "had", "do", "does", "did"]
        
        self.ignored_words.update(tech_words + literary_words + fantasy_words + common_words)
        if self.sym_spell:
            for w in self.ignored_words:
                self.sym_spell.create_dictionary_entry(w, 1)

    def load_custom_dictionary(self) -> None:
        self.dict_file = "custom_dictionary.txt"
        if os.path.exists(self.dict_file):
            with open(self.dict_file, 'r', encoding='utf-8') as f:
                words = f.read().splitlines()
                self.ignored_words.update(word.lower() for word in words if word.strip())
                if self.sym_spell:
                    for w in words:
                        self.sym_spell.create_dictionary_entry(w.lower(), 1)

    def highlightBlock(self, text: str) -> None:
        if not text or not text.strip(): return

        block_start = self.currentBlock().position()
        block_end = block_start + len(text)
        url_ranges = [(m.start(), m.end()) for m in self.url_pattern.finditer(text)]
        def in_url(pos): return any(start <= pos < end for start, end in url_ranges)

        for match in self.word_pattern.finditer(text):
            word = match.group(0)
            if not in_url(match.start()) and self.is_misspelled(word):
                self.setFormat(match.start(), match.end() - match.start(), self.spell_format)

        for rule in self.grammar_rules:
            for match in rule['pattern'].finditer(text):
                self.setFormat(match.start(), match.end() - match.start(), rule['format'])

        for start_offset, (length, msg, fmt_type) in self.nlp_errors.items():
            if block_start <= start_offset < block_end:
                local_start = start_offset - block_start
                fmt = self.grammar_format if fmt_type == "grammar" else self.passive_format
                self.setFormat(local_start, min(length, len(text) - local_start), fmt)


class TextEditWithSpellCheck(QTextEdit):
    """Custom QTextEdit with spell check, smart quotes, and autocorrect."""
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.highlighter = SpellGrammarHighlighter(self.document())
        self.line_highlight_enabled = False
        self.setViewportMargins(25, 20, 25, 20) 

        self.nlp_thread = NlpAnalysisThread()
        self.nlp_thread.errors_found.connect(self.highlighter.set_nlp_errors)
        
        self.nlp_timer = QTimer(self)
        self.nlp_timer.setSingleShot(True)
        self.nlp_timer.timeout.connect(self.run_nlp_analysis)
        self.textChanged.connect(self.nlp_timer.start)

    def keyPressEvent(self, event) -> None:
        # Smart Quotes & Quote Wrapping
        if event.text() in ('"', "'"):
            cursor = self.textCursor()
            if cursor.hasSelection():
                start = cursor.selectionStart()
                end = cursor.selectionEnd()
                cursor.beginEditBlock()
                cursor.setPosition(start)
                
                if event.text() == '"':
                    open_q, close_q = '“', '”'
                else:
                    open_q, close_q = '‘', '’'
                    
                cursor.insertText(open_q)
                cursor.setPosition(end + 1)  # +1 because we inserted one char
                cursor.insertText(close_q)
                cursor.endEditBlock()
                return
            else:
                char_before = self.document().characterAt(cursor.position() - 1)
                if event.text() == '"':
                    if cursor.atBlockStart() or char_before in (' ', '(', '\n', '\t'):
                        self.insertPlainText('“')
                    else:
                        self.insertPlainText('”')
                    return
                else: # Single quote
                    if cursor.atBlockStart() or char_before in (' ', '(', '\n', '\t'):
                        self.insertPlainText('‘')
                    else:
                        self.insertPlainText('’')
                    return

        super().keyPressEvent(event)

        # Autocorrect on space/enter
        if event.text() in (' ', '\t', '\n'):
            cursor = self.textCursor()
            if cursor.position() > 1:
                # Check the word before the just-typed space
                word_cursor = QTextCursor(cursor)
                word_cursor.movePosition(QTextCursor.MoveOperation.PreviousCharacter) # Skip the space we just typed
                word_cursor.movePosition(QTextCursor.MoveOperation.PreviousWord, QTextCursor.MoveMode.KeepAnchor) # Select the word
                
                word = word_cursor.selectedText().strip().lower()
                
                if word and ' ' not in word and '\t' not in word and '\n' not in word:
                    fix = self.highlighter.autocorrect_map.get(word)
                    if fix:
                        word_cursor.insertText(fix) # Replaces the selected word

    def run_nlp_analysis(self) -> None:
        if NLP:
            self.nlp_thread.set_text(self.toPlainText())
            if not self.nlp_thread.isRunning():
                self.nlp_thread.start()

    def contextMenuEvent(self, event) -> None:
        cursor = self.cursorForPosition(event.pos())
        temp_cursor = QTextCursor(cursor)
        temp_cursor.select(QTextCursor.SelectionType.WordUnderCursor)
        word = temp_cursor.selectedText().strip()
        clean_word = re.sub(r'[^A-Za-z\']', '', word)

        menu = self.createStandardContextMenu()

        if clean_word:
            # Spelling suggestions
            if self.highlighter.is_misspelled(clean_word):
                suggestions = self.highlighter.get_suggestions(clean_word)
                if suggestions:
                    menu.addSeparator()
                    menu.addAction(QAction("Spelling:", menu, enabled=False))
                    for suggestion in suggestions:
                        act = QAction(suggestion, menu)
                        act.triggered.connect(lambda checked, s=suggestion, c=temp_cursor: self.replace_word(c, s))
                        menu.addAction(act)
                    
                    menu.addSeparator()
                    add_dict_act = QAction(f"Add '{clean_word}' to Dictionary", menu)
                    add_dict_act.triggered.connect(lambda checked, w=clean_word: self.highlighter.add_to_permanent_dictionary(w))
                    menu.addAction(add_dict_act)
            
            # Thesaurus lookup
            synonyms = THESAURUS.get(clean_word.lower())
            if synonyms:
                menu.addSeparator()
                menu.addAction(QAction("Synonyms:", menu, enabled=False))
                for syn in synonyms:
                    act = QAction(syn, menu)
                    act.triggered.connect(lambda checked, s=syn, c=temp_cursor: self.replace_word(c, s))
                    menu.addAction(act)

        menu.exec(event.globalPos())

    def replace_word(self, cursor, new_word: str) -> None:
        cursor.insertText(new_word)
        self.setTextCursor(cursor)

    def set_line_highlight(self, enabled: bool) -> None:
        self.line_highlight_enabled = enabled
        self.update_extra_selections()

    def update_extra_selections(self) -> None:
        selections = []
        if self.line_highlight_enabled:
            cursor = self.textCursor()
            block = cursor.block()
            sel = QTextEdit.ExtraSelection()
            sel.format.setBackground(QColor(255, 240, 150, 80))
            sel.format.setProperty(QTextFormat.Property.FullWidthSelection, True)
            sel.cursor = QTextCursor(block)
            sel.cursor.movePosition(QTextCursor.MoveOperation.EndOfBlock, QTextCursor.MoveMode.KeepAnchor)
            selections.append(sel)
        self.setExtraSelections(selections)

    def find_next_spelling_error(self) -> bool:
        cursor = self.textCursor()
        doc = self.document()
        block = cursor.block()
        while block.isValid():
            text = block.text()
            offset = block.position()
            for m in self.highlighter.word_pattern.finditer(text):
                word = m.group(0)
                if self.highlighter.is_misspelled(word):
                    target = QTextCursor(doc)
                    target.setPosition(offset + m.start())
                    target.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, len(word))
                    self.setTextCursor(target)
                    self.ensureCursorVisible()
                    return True
            block = block.next()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.setTextCursor(cursor)
        return False

    def find_next_grammar_error(self) -> bool:
        cursor = self.textCursor()
        pos = cursor.position()
        if self.highlighter.nlp_errors:
            for start in sorted(self.highlighter.nlp_errors.keys()):
                if start > pos:
                    length, _msg, _ = self.highlighter.nlp_errors[start]
                    target = QTextCursor(self.document())
                    target.setPosition(start)
                    target.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, length)
                    self.setTextCursor(target)
                    self.ensureCursorVisible()
                    return True
        doc = self.document()
        block = cursor.block()
        while block.isValid():
            text = block.text()
            offset = block.position()
            for rule in self.highlighter.grammar_rules:
                if rule['category'] != 'grammar': continue
                for m in rule['pattern'].finditer(text):
                    if offset + m.start() > pos:
                        target = QTextCursor(doc)
                        target.setPosition(offset + m.start())
                        target.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, m.end() - m.start())
                        self.setTextCursor(target)
                        self.ensureCursorVisible()
                        return True
            block = block.next()
        return False

# --- Story Bible Dock Widget ---
class StoryBibleDock(QDockWidget):
    """Dock widget for storing characters, locations, and plot notes."""
    def __init__(self, parent=None) -> None:
        super().__init__("Story Bible & Notes", parent)
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.tabs = QTabWidget()
        self.char_edit = QTextEdit()
        self.loc_edit = QTextEdit()
        self.plot_edit = QTextEdit()
        self.notes_edit = QTextEdit()
        self.tabs.addTab(self.char_edit, "Characters")
        self.tabs.addTab(self.loc_edit, "Locations")
        self.tabs.addTab(self.plot_edit, "Plot")
        self.tabs.addTab(self.notes_edit, "Notes")
        self.setWidget(self.tabs)
        self.load_notes()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.save_notes)
        self.timer.start(10000)

    def load_notes(self) -> None:
        if os.path.exists("story_bible.json"):
            with open("story_bible.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.char_edit.setHtml(data.get("characters", ""))
                self.loc_edit.setHtml(data.get("locations", ""))
                self.plot_edit.setHtml(data.get("plot", ""))
                self.notes_edit.setHtml(data.get("notes", ""))

    def save_notes(self) -> None:
        data = {
            "characters": self.char_edit.toHtml(),
            "locations": self.loc_edit.toHtml(),
            "plot": self.plot_edit.toHtml(),
            "notes": self.notes_edit.toHtml()
        }
        with open("story_bible.json", 'w', encoding='utf-8') as f:
            json.dump(data, f)

# --- Main Application ---
class WordProcessor(QMainWindow):
    """Main application window."""
    def __init__(self) -> None:
        super().__init__()
        self.settings = QSettings("MyCompany", "PySideWordProcessor")
        self.setWindowTitle("PySide Word Processor Pro - Advanced NLP Engine")
        self.resize(1280, 800)
        self.path = None
        self.zoom_level = 0
        self.dictation_thread = None
        self.word_goal = 0
        self.quote_toggle = False
        self.sprint_active = False
        self.sprint_start_words = 0
        
        # Performance cache variables
        self.current_word_count = 0
        self.current_char_count = 0

        self.init_ui()
        self.init_tts()
        self.create_actions()
        self.create_toolbars()
        self.create_menus()
        self.create_status_bar()
        self.create_outline_dock()
        self.create_story_bible_dock()
        
        self.text_edit.setFontPointSize(12)
        self.update_character_formatting()
        
        # Throttle heavy operations for performance
        self.word_count_timer = QTimer(self)
        self.word_count_timer.setSingleShot(True)
        self.word_count_timer.timeout.connect(self._calculate_word_count)
        
        self.text_edit.textChanged.connect(self.on_text_changed)
        self.text_edit.cursorPositionChanged.connect(self.update_cursor_position)
        self.text_edit.cursorPositionChanged.connect(self.text_edit.update_extra_selections)
        self.update_zoom_label()
        
        self.load_settings()
        self.update_recent_files_menu()
        self.check_recovery_file()
        self.apply_modern_style()
        
        self.autosave_timer = QTimer(self)
        self.autosave_timer.timeout.connect(self.auto_save)
        self.autosave_timer.start(120000)
        
        self.outline_timer = QTimer(self)
        self.outline_timer.setSingleShot(True)
        self.outline_timer.timeout.connect(self.update_outline)
        self.outline_timer.start(1000)

        self.sprint_timer = QTimer(self)
        self.sprint_timer.timeout.connect(self.update_sprint_timer)

    def init_ui(self) -> None:
        self.text_edit = TextEditWithSpellCheck()
        self.text_edit.setFont(QFont("Segoe UI", 11))
        self.setCentralWidget(self.text_edit)
        self.text_edit.viewport().installEventFilter(self)
        self.text_edit.cursorPositionChanged.connect(self.update_character_formatting)
        self.text_edit.document().modificationChanged.connect(self.update_window_title)

    def init_tts(self) -> None:
        self.tts = QTextToSpeech()
        self.tts_state = QTextToSpeech.State.Ready
        self.tts.stateChanged.connect(self.on_tts_state_changed)

    def apply_modern_style(self) -> None:
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; }
            QToolBar { border: none; border-bottom: 1px solid #dcdcdc; background-color: #ffffff; spacing: 2px; padding: 3px; }
            QToolBar QToolButton { padding: 4px; border-radius: 4px; background: transparent; }
            QToolBar QToolButton:hover { background-color: #e0e0e0; }
            QToolBar QToolButton:checked { background-color: #d0d0d0; border: 1px solid #c0c0c0; }
            QStatusBar { border-top: 1px solid #dcdcdc; background-color: #ffffff; }
            QStatusBar QLabel { padding-left: 8px; padding-right: 8px; border-left: 1px solid #e0e0e0; }
            QDockWidget { border: none; }
            QDockWidget::title { background-color: #ffffff; border-bottom: 1px solid #dcdcdc; padding: 6px; font-weight: bold; }
            QListWidget { border: none; background-color: #ffffff; }
            QTabWidget::pane { border: 1px solid #dcdcdc; background-color: #ffffff; }
            QTabBar::tab { background: #f0f0f0; border: 1px solid #dcdcdc; padding: 6px 12px; margin-right: 2px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #ffffff; border-bottom: none; }
            QTextEdit { border: none; background-color: #ffffff; }
            QComboBox, QSpinBox, QLineEdit { border: 1px solid #c0c0c0; border-radius: 4px; padding: 2px 4px; background: white; }
            QComboBox:hover, QSpinBox:hover, QLineEdit:hover { border-color: #a0a0a0; }
            QPushButton { background-color: #0078d4; color: white; border: none; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #006cbd; }
            QPushButton:pressed { background-color: #005a9e; }
            QMenu { background-color: #ffffff; border: 1px solid #dcdcdc; }
            QMenu::item { padding: 6px 24px; border-radius: 4px; margin: 2px 4px; }
            QMenu::item:selected { background-color: #e0e0e0; }
            QMenu::separator { height: 1px; background: #e0e0e0; margin: 4px 8px; }
            QProgressBar { border: 1px solid #c0c0c0; border-radius: 4px; text-align: center; background: white; }
            QProgressBar::chunk { background-color: #0078d4; border-radius: 3px; }
            QScrollBar:vertical { border: none; background: #f0f0f0; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #c0c0c0; min-height: 20px; border-radius: 4px; }
            QScrollBar::handle:vertical:hover { background: #a0a0a0; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }
            QScrollBar:horizontal { border: none; background: #f0f0f0; height: 10px; margin: 0px; }
            QScrollBar::handle:horizontal { background: #c0c0c0; min-width: 20px; border-radius: 4px; }
            QScrollBar::handle:horizontal:hover { background: #a0a0a0; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0px; }
        """)

    def load_settings(self) -> None:
        geometry = self.settings.value("geometry")
        if geometry: self.restoreGeometry(geometry)
        is_dark = self.settings.value("dark_mode", False, type=bool)
        self.dark_mode_action.setChecked(is_dark)
        self.toggle_dark_mode(is_dark)
        self.word_goal = self.settings.value("word_goal", 0, type=int)
        if self.word_goal > 0:
            self.goal_progress_bar.show()
            self.goal_label.setText(f"Goal: {self.word_goal}")

    def save_settings(self) -> None:
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("dark_mode", self.dark_mode_action.isChecked())
        self.settings.setValue("word_goal", self.word_goal)
        if self.path: self.add_to_recent_files(self.path)

    def update_window_title(self, modified: bool) -> None:
        title = "PySide Word Processor Pro"
        if self.path: title = f"{os.path.basename(self.path)} - {title}"
        else: title = f"Untitled - {title}"
        if modified: title = f"*{title}"
        self.setWindowTitle(title)

    def create_actions(self) -> None:
        self.new_action = QAction("&New", self); self.new_action.setShortcut(QKeySequence.StandardKey.New); self.new_action.triggered.connect(self.new_document)
        self.open_action = QAction("&Open...", self); self.open_action.setShortcut(QKeySequence.StandardKey.Open); self.open_action.triggered.connect(self.open_document)
        self.save_action = QAction("&Save", self); self.save_action.setShortcut(QKeySequence.StandardKey.Save); self.save_action.triggered.connect(self.save_document)
        self.save_as_action = QAction("Save &As...", self); self.save_as_action.setShortcut(QKeySequence.StandardKey.SaveAs); self.save_as_action.triggered.connect(self.save_as_document)
        self.export_pdf_action = QAction("Export to &PDF...", self); self.export_pdf_action.triggered.connect(self.export_pdf)
        self.export_manuscript_action = QAction("Export Standard &Manuscript...", self); self.export_manuscript_action.triggered.connect(self.export_manuscript)
        self.print_action = QAction("&Print...", self); self.print_action.setShortcut(QKeySequence.StandardKey.Print); self.print_action.triggered.connect(self.print_document)
        self.print_preview_action = QAction("Print Pre&view...", self); self.print_preview_action.triggered.connect(self.print_preview)
        self.page_setup_action = QAction("Page Set&up...", self); self.page_setup_action.triggered.connect(self.page_setup)
        self.exit_action = QAction("E&xit", self); self.exit_action.setShortcut(QKeySequence.StandardKey.Quit); self.exit_action.triggered.connect(self.close)

        self.undo_action = QAction("&Undo", self); self.undo_action.setShortcut(QKeySequence.StandardKey.Undo); self.undo_action.triggered.connect(self.text_edit.undo)
        self.redo_action = QAction("&Redo", self); self.redo_action.setShortcut(QKeySequence.StandardKey.Redo); self.redo_action.triggered.connect(self.text_edit.redo)
        self.cut_action = QAction("Cu&t", self); self.cut_action.setShortcut(QKeySequence.StandardKey.Cut); self.cut_action.triggered.connect(self.text_edit.cut)
        self.copy_action = QAction("&Copy", self); self.copy_action.setShortcut(QKeySequence.StandardKey.Copy); self.copy_action.triggered.connect(self.text_edit.copy)
        self.paste_action = QAction("&Paste", self); self.paste_action.setShortcut(QKeySequence.StandardKey.Paste); self.paste_action.triggered.connect(self.text_edit.paste)
        self.find_action = QAction("&Find...", self); self.find_action.setShortcut(QKeySequence.StandardKey.Find); self.find_action.triggered.connect(self.open_find_dialog)
        
        self.next_spell_action = QAction("Next Spelling Error", self); self.next_spell_action.setShortcut(QKeySequence("F7")); self.next_spell_action.triggered.connect(self.text_edit.find_next_spelling_error)
        self.next_grammar_action = QAction("Next Grammar Error", self); self.next_grammar_action.setShortcut(QKeySequence("Shift+F7")); self.next_grammar_action.triggered.connect(self.text_edit.find_next_grammar_error)
        self.proofreading_action = QAction("Proofreading &Report...", self); self.proofreading_action.setShortcut(QKeySequence("Ctrl+Shift+F7")); self.proofreading_action.triggered.connect(self.open_proofreading_report)
        self.dict_manager_action = QAction("Manage Custom &Dictionary...", self); self.dict_manager_action.triggered.connect(self.open_dict_manager)
        self.readability_action = QAction("Readability &Analysis...", self); self.readability_action.triggered.connect(self.show_readability)
        self.academic_mode_action = QAction("&Academic Mode", self); self.academic_mode_action.setCheckable(True); self.academic_mode_action.toggled.connect(self.toggle_academic_mode)

        self.bold_action = QAction(create_text_icon("B"), "&Bold", self); self.bold_action.setShortcut(QKeySequence.StandardKey.Bold); self.bold_action.setCheckable(True); self.bold_action.triggered.connect(self.toggle_bold)
        self.italic_action = QAction(create_text_icon("I", bold=False), "&Italic", self); self.italic_action.setShortcut(QKeySequence.StandardKey.Italic); self.italic_action.setCheckable(True); self.italic_action.triggered.connect(self.toggle_italic)
        self.underline_action = QAction(create_text_icon("U"), "&Underline", self); self.underline_action.setShortcut(QKeySequence.StandardKey.Underline); self.underline_action.setCheckable(True); self.underline_action.triggered.connect(self.toggle_underline)
        self.strike_action = QAction(create_text_icon("S"), "&Strikethrough", self); self.strike_action.setCheckable(True); self.strike_action.triggered.connect(self.toggle_strikethrough)
        self.color_action = QAction("Text &Color...", self); self.color_action.triggered.connect(self.choose_color)
        self.highlight_action = QAction("&Highlight...", self); self.highlight_action.triggered.connect(self.choose_highlight)

        self.bullet_list_action = QAction(create_text_icon("•", font_size=20), "&Bullet List", self); self.bullet_list_action.setCheckable(True); self.bullet_list_action.triggered.connect(self.toggle_bullet_list)
        self.numbered_list_action = QAction(create_text_icon("1.", font_size=14), "&Numbered List", self); self.numbered_list_action.setCheckable(True); self.numbered_list_action.triggered.connect(self.toggle_numbered_list)
        self.indent_inc_action = QAction(create_text_icon("→", bold=False), "Increase Indent", self); self.indent_inc_action.triggered.connect(self.increase_indent)
        self.indent_dec_action = QAction(create_text_icon("←", bold=False), "Decrease Indent", self); self.indent_dec_action.triggered.connect(self.decrease_indent)
        
        self.insert_image_action = QAction("Insert &Image...", self); self.insert_image_action.triggered.connect(self.insert_image)
        self.insert_link_action = QAction("Insert &Hyperlink...", self); self.insert_link_action.setShortcut(QKeySequence("Ctrl+K")); self.insert_link_action.triggered.connect(self.insert_hyperlink)
        self.page_break_action = QAction("Insert Page &Break", self); self.page_break_action.setShortcut(QKeySequence("Ctrl+Return")); self.page_break_action.triggered.connect(self.insert_page_break)
        self.insert_table_action = QAction("Insert &Table...", self); self.insert_table_action.triggered.connect(self.insert_table)
        
        self.sym_em_dash = QAction("Em Dash (—)", self); self.sym_em_dash.triggered.connect(lambda: self.text_edit.insertPlainText("—"))
        self.sym_en_dash = QAction("En Dash (–)", self); self.sym_en_dash.triggered.connect(lambda: self.text_edit.insertPlainText("–"))
        self.sym_ellipsis = QAction("Ellipsis (…)", self); self.sym_ellipsis.triggered.connect(lambda: self.text_edit.insertPlainText("…"))
        self.sym_copyright = QAction("Copyright (©)", self); self.sym_copyright.triggered.connect(lambda: self.text_edit.insertPlainText("©"))

        self.align_left_action = QAction("Align &Left", self); self.align_left_action.setCheckable(True); self.align_left_action.triggered.connect(lambda: self.set_alignment(Qt.AlignmentFlag.AlignLeft))
        self.align_center_action = QAction("Align &Center", self); self.align_center_action.setCheckable(True); self.align_center_action.triggered.connect(lambda: self.set_alignment(Qt.AlignmentFlag.AlignCenter))
        self.align_right_action = QAction("Align &Right", self); self.align_right_action.setCheckable(True); self.align_right_action.triggered.connect(lambda: self.set_alignment(Qt.AlignmentFlag.AlignRight))
        self.align_justify_action = QAction("Align &Justify", self); self.align_justify_action.setCheckable(True); self.align_justify_action.triggered.connect(lambda: self.set_alignment(Qt.AlignmentFlag.AlignJustify))
        
        self.zoom_in_action = QAction("Zoom &In", self); self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn); self.zoom_in_action.triggered.connect(self.zoom_in)
        self.zoom_out_action = QAction("Zoom &Out", self); self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut); self.zoom_out_action.triggered.connect(self.zoom_out)
        self.dark_mode_action = QAction("&Dark Mode", self); self.dark_mode_action.setCheckable(True); self.dark_mode_action.toggled.connect(self.toggle_dark_mode)
        self.line_highlight_action = QAction("&Reading Ruler", self); self.line_highlight_action.setCheckable(True); self.line_highlight_action.toggled.connect(self.text_edit.set_line_highlight)

        self.speak_action = QAction("&Read Aloud", self); self.speak_action.setShortcut(QKeySequence("Ctrl+R")); self.speak_action.triggered.connect(self.read_aloud)
        self.stop_speak_action = QAction("&Stop Reading", self); self.stop_speak_action.triggered.connect(self.stop_reading)
        self.pause_speak_action = QAction("&Pause", self); self.pause_speak_action.triggered.connect(self.pause_reading)
        self.dictate_action = QAction(create_text_icon("🎙", font_size=16, bold=False), "&Dictate", self); self.dictate_action.setCheckable(True)
        if not SR_AVAILABLE: self.dictate_action.setEnabled(False)
        else: self.dictate_action.toggled.connect(self.toggle_dictation)

        self.focus_mode_action = QAction("&Focus Mode", self); self.focus_mode_action.setCheckable(True); self.focus_mode_action.setShortcuts([QKeySequence("F11"), QKeySequence("Esc")]); self.focus_mode_action.toggled.connect(self.toggle_focus_mode)
        self.stats_action = QAction("Document Statistic&s...", self); self.stats_action.triggered.connect(self.show_stats)
        self.freq_action = QAction("Word &Frequency Analyzer...", self); self.freq_action.triggered.connect(self.show_word_frequency)

        self.insert_chapter_action = QAction("Insert &Chapter Break", self); self.insert_chapter_action.triggered.connect(self.insert_chapter_break)
        self.insert_scene_action = QAction("Insert &Scene Break", self); self.insert_scene_action.triggered.connect(self.insert_scene_break)
        self.name_fantasy_action = QAction("Insert Fantasy Name", self); self.name_fantasy_action.triggered.connect(lambda: self.insert_name("Fantasy"))
        self.name_modern_action = QAction("Insert Modern Name", self); self.name_modern_action.triggered.connect(lambda: self.insert_name("Modern"))
        
        self.sp_scene_action = QAction("Format: &Scene Heading", self); self.sp_scene_action.triggered.connect(lambda: self.format_screenplay("Scene Heading"))
        self.sp_action_fmt_action = QAction("Format: &Action", self); self.sp_action_fmt_action.triggered.connect(lambda: self.format_screenplay("Action"))
        self.sp_char_action = QAction("Format: &Character Name", self); self.sp_char_action.triggered.connect(lambda: self.format_screenplay("Character"))
        self.sp_dial_action = QAction("Format: &Dialogue", self); self.sp_dial_action.triggered.connect(lambda: self.format_screenplay("Dialogue"))
        self.sp_paren_action = QAction("Format: &Parenthetical", self); self.sp_paren_action.triggered.connect(lambda: self.format_screenplay("Parenthetical"))
        self.sp_trans_action = QAction("Format: &Transition", self); self.sp_trans_action.triggered.connect(lambda: self.format_screenplay("Transition"))

    def create_toolbars(self) -> None:
        self.file_toolbar = self.addToolBar("File"); self.file_toolbar.setMovable(True); self.file_toolbar.setIconSize(QSize(20, 20))
        self.file_toolbar.addAction(self.new_action); self.file_toolbar.addAction(self.open_action); self.file_toolbar.addAction(self.save_action)
        self.file_toolbar.addAction(self.print_action); self.file_toolbar.addAction(self.print_preview_action)
        self.file_toolbar.addSeparator()
        self.file_toolbar.addAction(self.insert_image_action); self.file_toolbar.addAction(self.insert_link_action)
        self.file_toolbar.addAction(self.page_break_action); self.file_toolbar.addAction(self.insert_table_action)
        self.file_toolbar.addSeparator()
        self.file_toolbar.addAction(self.zoom_out_action); self.file_toolbar.addAction(self.zoom_in_action)
        self.file_toolbar.addWidget(QLabel(" Zoom: ")); self.zoom_label = QLabel("100%"); self.file_toolbar.addWidget(self.zoom_label)

        self.format_toolbar = self.addToolBar("Format"); self.format_toolbar.setMovable(True); self.format_toolbar.setIconSize(QSize(20, 20))
        self.font_combo = QFontComboBox(); self.font_combo.currentFontChanged.connect(self.set_font_family); self.format_toolbar.addWidget(self.font_combo)
        self.font_size_combo = QComboBox(); self.font_size_combo.setEditable(True)
        sizes = [8, 9, 10, 11, 12, 14, 16, 18, 20, 24, 28, 32, 48, 72]
        self.font_size_combo.addItems([str(s) for s in sizes]); self.font_size_combo.setCurrentText("12")
        self.font_size_combo.currentTextChanged.connect(self.set_font_size); self.format_toolbar.addWidget(self.font_size_combo)
        self.format_toolbar.addWidget(QLabel(" Lang: "))
        self.dict_lang_combo = QComboBox()
        self.dict_lang_combo.addItems(["English (US)", "English (UK)", "English (AU)"])
        self.dict_lang_combo.currentIndexChanged.connect(self.change_dict_language)
        self.format_toolbar.addWidget(self.dict_lang_combo)
        self.format_toolbar.addSeparator()
        self.format_toolbar.addAction(self.bold_action); self.format_toolbar.addAction(self.italic_action)
        self.format_toolbar.addAction(self.underline_action); self.format_toolbar.addAction(self.strike_action)
        self.format_toolbar.addAction(self.color_action); self.format_toolbar.addAction(self.highlight_action)
        self.format_toolbar.addSeparator()
        self.format_toolbar.addAction(self.align_left_action); self.format_toolbar.addAction(self.align_center_action)
        self.format_toolbar.addAction(self.align_right_action); self.format_toolbar.addAction(self.align_justify_action)
        self.format_toolbar.addSeparator()
        self.format_toolbar.addAction(self.bullet_list_action); self.format_toolbar.addAction(self.numbered_list_action)
        self.format_toolbar.addAction(self.indent_dec_action); self.format_toolbar.addAction(self.indent_inc_action)
        self.format_toolbar.addWidget(QLabel(" Spacing: "))
        self.line_spacing_combo = QComboBox(); self.line_spacing_combo.addItems(["1.0", "1.15", "1.5", "2.0", "2.5", "3.0"])
        self.line_spacing_combo.currentTextChanged.connect(self.set_line_spacing); self.format_toolbar.addWidget(self.line_spacing_combo)
        
        self.speech_toolbar = self.addToolBar("Speech"); self.speech_toolbar.setMovable(True); self.speech_toolbar.setIconSize(QSize(20, 20))
        self.speech_toolbar.addAction(self.dictate_action); self.speech_toolbar.addSeparator()
        self.speech_toolbar.addAction(self.speak_action); self.speech_toolbar.addAction(self.pause_speak_action); self.speech_toolbar.addAction(self.stop_speak_action)
        self.speech_toolbar.addSeparator(); self.speech_toolbar.addWidget(QLabel(" Voice: "))
        self.voice_combo = QComboBox(); self.voice_combo.currentIndexChanged.connect(self.set_tts_voice); self.speech_toolbar.addWidget(self.voice_combo)
        self.speech_toolbar.addWidget(QLabel(" Rate: "))
        self.rate_slider = QSlider(Qt.Orientation.Horizontal); self.rate_slider.setRange(-100, 100); self.rate_slider.setValue(0)
        self.rate_slider.setFixedWidth(100); self.rate_slider.valueChanged.connect(self.set_tts_rate); self.speech_toolbar.addWidget(self.rate_slider)

        self.story_toolbar = self.addToolBar("Story Tools"); self.story_toolbar.setMovable(True); self.story_toolbar.setIconSize(QSize(20, 20))
        self.story_toolbar.addAction(self.insert_chapter_action); self.story_toolbar.addAction(self.insert_scene_action)
        self.story_toolbar.addSeparator()
        self.story_toolbar.addAction(self.sp_scene_action); self.story_toolbar.addAction(self.sp_char_action)
        self.story_toolbar.addAction(self.sp_dial_action); self.story_toolbar.addAction(self.sp_paren_action); self.story_toolbar.addAction(self.sp_trans_action)
        self.story_toolbar.addSeparator()
        self.story_toolbar.addAction(self.focus_mode_action); self.story_toolbar.addAction(self.stats_action); self.story_toolbar.addAction(self.freq_action); self.story_toolbar.addAction(self.readability_action)
        self.story_toolbar.addSeparator()
        self.story_toolbar.addAction(self.next_spell_action); self.story_toolbar.addAction(self.next_grammar_action); self.story_toolbar.addAction(self.proofreading_action)

        self.populate_voices()
        self.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)

    def create_menus(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        file_menu.addAction(self.new_action); file_menu.addAction(self.open_action)
        self.recent_files_menu = file_menu.addMenu("Open &Recent")
        file_menu.addAction(self.save_action); file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.page_setup_action); file_menu.addAction(self.print_preview_action); file_menu.addAction(self.print_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_pdf_action); file_menu.addAction(self.export_manuscript_action)
        file_menu.addSeparator(); file_menu.addAction(self.exit_action)

        edit_menu = menubar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action); edit_menu.addAction(self.redo_action); edit_menu.addSeparator()
        edit_menu.addAction(self.cut_action); edit_menu.addAction(self.copy_action); edit_menu.addAction(self.paste_action)
        edit_menu.addSeparator()
        edit_menu.addAction(self.find_action); edit_menu.addAction(self.next_spell_action); edit_menu.addAction(self.next_grammar_action)
        edit_menu.addAction(self.proofreading_action)
        edit_menu.addSeparator(); edit_menu.addAction(self.dict_manager_action)
        edit_menu.addSeparator(); edit_menu.addAction(self.academic_mode_action)

        format_menu = menubar.addMenu("&Format")
        format_menu.addAction(self.bold_action); format_menu.addAction(self.italic_action); format_menu.addAction(self.underline_action)
        format_menu.addAction(self.strike_action); format_menu.addAction(self.color_action); format_menu.addAction(self.highlight_action)
        format_menu.addSeparator()
        format_menu.addAction(self.bullet_list_action); format_menu.addAction(self.numbered_list_action)
        format_menu.addAction(self.indent_dec_action); format_menu.addAction(self.indent_inc_action)
        format_menu.addSeparator()
        format_menu.addAction(self.align_left_action); format_menu.addAction(self.align_center_action)
        format_menu.addAction(self.align_right_action); format_menu.addAction(self.align_justify_action)
        
        insert_menu = menubar.addMenu("&Insert")
        insert_menu.addAction(self.insert_image_action); insert_menu.addAction(self.insert_link_action)
        insert_menu.addAction(self.page_break_action); insert_menu.addAction(self.insert_table_action)
        sym_menu = insert_menu.addMenu("&Typographic Symbols")
        sym_menu.addAction(self.sym_em_dash); sym_menu.addAction(self.sym_en_dash)
        sym_menu.addAction(self.sym_ellipsis); sym_menu.addAction(self.sym_copyright)
        name_menu = insert_menu.addMenu("&Quick Name Generator")
        name_menu.addAction(self.name_fantasy_action); name_menu.addAction(self.name_modern_action)
        
        story_menu = menubar.addMenu("&Story Tools")
        story_menu.addAction(self.focus_mode_action); story_menu.addAction(self.stats_action); story_menu.addAction(self.freq_action); story_menu.addAction(self.readability_action)
        story_menu.addSeparator()
        story_menu.addAction(self.insert_chapter_action); story_menu.addAction(self.insert_scene_action)
        sp_menu = story_menu.addMenu("&Screenplay Format")
        sp_menu.addAction(self.sp_scene_action); sp_menu.addAction(self.sp_action_fmt_action); sp_menu.addAction(self.sp_char_action)
        sp_menu.addAction(self.sp_paren_action); sp_menu.addAction(self.sp_dial_action); sp_menu.addAction(self.sp_trans_action)
        
        speech_menu = menubar.addMenu("&Speech")
        speech_menu.addAction(self.dictate_action); speech_menu.addSeparator()
        speech_menu.addAction(self.speak_action); speech_menu.addAction(self.pause_speak_action); speech_menu.addAction(self.stop_speak_action)
        
        view_menu = menubar.addMenu("&View")
        view_menu.addAction(self.zoom_in_action); view_menu.addAction(self.zoom_out_action); view_menu.addAction(self.line_highlight_action)
        view_menu.addSeparator()
        
        toolbars_menu = view_menu.addMenu("&Toolbars")
        for tb in [self.file_toolbar, self.format_toolbar, self.speech_toolbar, self.story_toolbar]:
            action = QAction(tb.windowTitle(), self, checkable=True)
            action.setChecked(True)
            action.triggered.connect(lambda checked, t=tb: t.setVisible(checked))
            toolbars_menu.addAction(action)

        view_menu.addSeparator()
        view_menu.addAction(self.dark_mode_action)

    def create_status_bar(self) -> None:
        self.statusBar().showMessage("Ready")
        self.sprint_btn = QPushButton("Start Sprint"); self.sprint_btn.clicked.connect(self.start_sprint); self.statusBar().addPermanentWidget(self.sprint_btn)
        self.goal_label = QPushButton("Set Goal"); self.goal_label.clicked.connect(self.set_word_goal); self.statusBar().addPermanentWidget(self.goal_label)
        self.goal_progress_bar = QProgressBar(); self.goal_progress_bar.setFixedWidth(150); self.goal_progress_bar.setFixedHeight(15); self.goal_progress_bar.hide()
        self.statusBar().addPermanentWidget(self.goal_progress_bar)
        self.stats_label = QLabel("Words: 0 | Chars: 0 | Ln: 1, Col: 1"); self.statusBar().addPermanentWidget(self.stats_label)

    def create_outline_dock(self) -> None:
        self.outline_dock = QDockWidget("Document Outline", self)
        self.outline_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self.outline_list = QListWidget(); self.outline_list.itemClicked.connect(self.navigate_to_outline_item)
        self.outline_dock.setWidget(self.outline_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.outline_dock)

    def create_story_bible_dock(self) -> None:
        self.story_bible_dock = StoryBibleDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.story_bible_dock)

    def add_to_recent_files(self, file_path: str) -> None:
        if not file_path: return
        recent_files = self.settings.value("recent_files", [], type=list)
        if file_path in recent_files: recent_files.remove(file_path)
        recent_files.insert(0, file_path); recent_files = recent_files[:5]
        self.settings.setValue("recent_files", recent_files); self.update_recent_files_menu()

    def update_recent_files_menu(self) -> None:
        self.recent_files_menu.clear()
        recent_files = self.settings.value("recent_files", [], type=list)
        if not recent_files:
            empty_action = QAction("No recent files", self); empty_action.setEnabled(False)
            self.recent_files_menu.addAction(empty_action); return
        for file_path in recent_files:
            action = QAction(os.path.basename(file_path), self); action.setToolTip(file_path)
            action.triggered.connect(lambda checked, p=file_path: self.open_recent_file(p))
            self.recent_files_menu.addAction(action)

    def open_recent_file(self, file_path: str) -> None:
        if not self.maybe_save(): return
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.txt'): self.text_edit.setPlainText(f.read())
                else: self.text_edit.setHtml(f.read())
            self.path = file_path; self.text_edit.document().setModified(False); self.update_window_title(False)
        else: QMessageBox.warning(self, "Error", "File not found.")

    def auto_save(self) -> None:
        if not self.text_edit.document().isModified(): return
        try:
            with open("recovery.html", 'w', encoding='utf-8') as f: f.write(self.text_edit.toHtml())
            if self.path and os.path.exists(self.path):
                tmp = self.path + ".tmp"
                with open(tmp, 'w', encoding='utf-8') as f:
                    if self.path.endswith('.txt'): f.write(self.text_edit.toPlainText())
                    else: f.write(self.text_edit.toHtml())
                os.replace(tmp, self.path)
                self.text_edit.document().setModified(False); self.update_window_title(False)
                self.statusBar().showMessage("Auto-saved", 2000)
        except OSError: pass

    def check_recovery_file(self) -> None:
        if os.path.exists("recovery.html"):
            ret = QMessageBox.question(self, "Recover Document", "An unsaved document was found. Would you like to recover it?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret == QMessageBox.StandardButton.Yes:
                with open("recovery.html", 'r', encoding='utf-8') as f: self.text_edit.setHtml(f.read())
                self.text_edit.document().setModified(True); self.update_window_title(True)
            else:
                os.remove("recovery.html")

    def new_document(self) -> None:
        if self.maybe_save():
            self.text_edit.clear(); self.path = None; self.text_edit.document().setModified(False); self.update_window_title(False)

    def open_document(self) -> None:
        if not self.maybe_save(): return
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Document", "", "HTML Files (*.html *.htm);;Text Files (*.txt);;All Files (*)")
        if file_path:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.txt'): self.text_edit.setPlainText(f.read())
                else: self.text_edit.setHtml(f.read())
            self.path = file_path; self.text_edit.document().setModified(False); self.update_window_title(False)
            self.add_to_recent_files(file_path)

    def save_document(self) -> bool:
        if not self.path: return self.save_as_document()
        try:
            tmp = self.path + ".tmp"
            with open(tmp, 'w', encoding='utf-8') as f:
                if self.path.endswith('.txt'): f.write(self.text_edit.toPlainText())
                else: f.write(self.text_edit.toHtml())
            os.replace(tmp, self.path)
            self.text_edit.document().setModified(False)
            self.add_to_recent_files(self.path)
            if os.path.exists("recovery.html"): os.remove("recovery.html")
            return True
        except OSError as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save file:\n{e}")
            return False

    def save_as_document(self) -> bool:
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Document", "Untitled.html", "HTML Files (*.html *.htm);;Text Files (*.txt)")
        if file_path:
            self.path = file_path; self.update_window_title(True); return self.save_document()
        return False

    def export_pdf(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Export to PDF", "Untitled.pdf", "PDF Files (*.pdf)")
        if file_path:
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution); printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat); printer.setOutputFileName(file_path)
            self.text_edit.document().print_(printer); self.statusBar().showMessage("PDF exported successfully.", 3000)
            QApplication.restoreOverrideCursor()

    def export_manuscript(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Standard Manuscript", "Manuscript.pdf", "PDF Files (*.pdf)")
        if not file_path: return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        doc = QTextDocument(); plain_text = self.text_edit.toPlainText()
        font = QFont("Times New Roman", 12); doc.setDefaultFont(font)
        cursor = QTextCursor(doc); fmt = QTextBlockFormat()
        fmt.setLineHeight(200, QTextBlockFormat.LineHeightTypes.ProportionalHeight)
        first = True
        for paragraph in plain_text.split('\n'):
            if not first: cursor.insertBlock(fmt)
            first = False
            block_fmt = QTextBlockFormat(fmt)
            block_fmt.setTextIndent(0 if not paragraph.strip() else 20)
            cursor.setBlockFormat(block_fmt)
            cursor.insertText(paragraph)
        printer = QPrinter(QPrinter.PrinterMode.HighResolution); printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat); printer.setOutputFileName(file_path)
        printer.setPageSize(QPageSize(QPageSize.PageSizeId.Letter)); printer.setPageMargins(QMarginsF(1.0, 1.0, 1.0, 1.0), QPageLayout.Unit.Inch)
        doc.print_(printer); self.statusBar().showMessage("Manuscript exported successfully.", 3000)
        QApplication.restoreOverrideCursor()

    def print_document(self) -> None:
        printer = QPrinter(QPrinter.PrinterMode.HighResolution); dialog = QPrintDialog(printer, self)
        if dialog.exec() == QPrintDialog.DialogCode.Accepted: self.text_edit.document().print_(printer)

    def print_preview(self) -> None:
        printer = QPrinter(QPrinter.PrinterMode.HighResolution); preview = QPrintPreviewDialog(printer, self)
        preview.paintRequested.connect(self._render_preview)
        preview.exec()

    def _render_preview(self, printer) -> None:
        self.text_edit.document().print_(printer)

    def page_setup(self) -> None:
        printer = QPrinter(QPrinter.PrinterMode.HighResolution); dialog = QPageSetupDialog(printer, self)
        if dialog.exec() == QPageSetupDialog.DialogCode.Accepted: self.text_edit.document().setPageSize(printer.pageRect().size())

    def maybe_save(self) -> bool:
        if self.text_edit.document().isModified():
            ret = QMessageBox.warning(self, "Word Processor", "The document has been modified.\nDo you want to save your changes?", QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if ret == QMessageBox.StandardButton.Save: return self.save_document()
            elif ret == QMessageBox.StandardButton.Cancel: return False
        return True

    def open_find_dialog(self) -> None:
        self.find_dialog = FindReplaceDialog(self.text_edit, self); self.find_dialog.show()
        
    def open_dict_manager(self) -> None:
        dialog = DictionaryManagerDialog(self.text_edit.highlighter, self); dialog.exec()

    def open_proofreading_report(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        dialog = ProofreadingDialog(self.text_edit, self.text_edit.highlighter, self); dialog.exec()
        QApplication.restoreOverrideCursor()

    # --- Formatting & Tools ---
    def toggle_bold(self) -> None: self.text_edit.setFontWeight(QFont.Weight.Bold if self.bold_action.isChecked() else QFont.Weight.Normal)
    def toggle_italic(self) -> None: self.text_edit.setFontItalic(self.italic_action.isChecked())
    def toggle_underline(self) -> None: self.text_edit.setFontUnderline(self.underline_action.isChecked())
    def toggle_strikethrough(self) -> None:
        fmt = self.text_edit.currentCharFormat(); fmt.setFontStrikeOut(self.strike_action.isChecked()); self.text_edit.setCurrentCharFormat(fmt)
    def choose_color(self) -> None:
        color = QColorDialog.getColor(self.text_edit.textColor(), self, "Select Text Color")
        if color.isValid(): self.text_edit.setTextColor(color)
    def choose_highlight(self) -> None:
        color = QColorDialog.getColor(Qt.GlobalColor.yellow, self, "Select Highlight Color")
        if color.isValid():
            fmt = self.text_edit.currentCharFormat(); fmt.setBackground(color); self.text_edit.setCurrentCharFormat(fmt)
    def set_font_family(self, font) -> None: self.text_edit.setCurrentFont(font)
    def set_font_size(self, size_str: str) -> None:
        try: size = float(size_str); self.text_edit.setFontPointSize(size)
        except ValueError: pass
    def set_alignment(self, alignment) -> None: self.text_edit.setAlignment(alignment)
    def set_line_spacing(self, text: str) -> None:
        try:
            factor = float(text); cursor = self.text_edit.textCursor(); block_fmt = cursor.blockFormat()
            block_fmt.setLineHeight(factor * 100, QTextBlockFormat.LineHeightTypes.ProportionalHeight); cursor.setBlockFormat(block_fmt); self.text_edit.setTextCursor(cursor)
        except ValueError: pass
    def toggle_bullet_list(self) -> None:
        cursor = self.text_edit.textCursor(); list_fmt = QTextListFormat()
        if cursor.currentList():
            if cursor.currentList().format().style() == QTextListFormat.Style.ListDisc: list_fmt.setStyle(QTextListFormat.Style.ListStyleUndefined)
            else: list_fmt.setStyle(QTextListFormat.Style.ListDisc)
        else: list_fmt.setStyle(QTextListFormat.Style.ListDisc)
        cursor.createList(list_fmt)
    def toggle_numbered_list(self) -> None:
        cursor = self.text_edit.textCursor(); list_fmt = QTextListFormat()
        if cursor.currentList():
            if cursor.currentList().format().style() == QTextListFormat.Style.ListDecimal: list_fmt.setStyle(QTextListFormat.Style.ListStyleUndefined)
            else: list_fmt.setStyle(QTextListFormat.Style.ListDecimal)
        else: list_fmt.setStyle(QTextListFormat.Style.ListDecimal)
        cursor.createList(list_fmt)
    def increase_indent(self) -> None:
        cursor = self.text_edit.textCursor(); block_fmt = cursor.blockFormat(); block_fmt.setIndent(block_fmt.indent() + 1)
        cursor.setBlockFormat(block_fmt); self.text_edit.setTextCursor(cursor)
    def decrease_indent(self) -> None:
        cursor = self.text_edit.textCursor(); block_fmt = cursor.blockFormat()
        if block_fmt.indent() > 0:
            block_fmt.setIndent(block_fmt.indent() - 1); cursor.setBlockFormat(block_fmt); self.text_edit.setTextCursor(cursor)
    def insert_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Insert Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.gif)")
        if file_path:
            image = QPixmap(file_path)
            if not image.isNull():
                doc_width = self.text_edit.viewport().width() - 20
                if image.width() > doc_width: image = image.scaledToWidth(doc_width, Qt.TransformationMode.SmoothTransformation)
                url = QUrl.fromLocalFile(file_path); self.text_edit.document().addResource(QTextDocument.ResourceType.ImageResource, url, image.toImage())
                self.text_edit.insertHtml(f'<img src="{url.toString()}" />')
            else: QMessageBox.warning(self, "Error", "Could not load image file.")
    def insert_hyperlink(self) -> None:
        cursor = self.text_edit.textCursor(); selected_text = cursor.selectedText()
        url, ok = QInputDialog.getText(self, "Insert Hyperlink", "URL:", text="https://")
        if ok and url:
            if not selected_text: selected_text = url
            self.text_edit.insertHtml(f'<a href="{url}">{selected_text}</a>')
    def insert_page_break(self) -> None: self.text_edit.insertPlainText("\f")
    def insert_table(self) -> None:
        rows, ok = QInputDialog.getInt(self, "Insert Table", "Number of rows:", 3, 1, 50)
        if not ok: return
        cols, ok = QInputDialog.getInt(self, "Insert Table", "Number of columns:", 3, 1, 10)
        if not ok: return
        cursor = self.text_edit.textCursor(); fmt = QTextTableFormat(); fmt.setCellSpacing(2); fmt.setCellPadding(5); fmt.setBorder(1)
        cursor.insertTable(rows, cols, fmt)

    def toggle_focus_mode(self, checked: bool) -> None:
        self.menuBar().setVisible(not checked); self.outline_dock.setVisible(not checked)
        self.story_bible_dock.setVisible(not checked); self.statusBar().setVisible(not checked)
        for tb in self.findChildren(QToolBar): tb.setVisible(not checked)
        if checked:
            bg_color = "#1e1e1e" if self.dark_mode_action.isChecked() else "#fafafa"
            self.text_edit.setStyleSheet(f"QTextEdit {{ margin: 50px; background-color: {bg_color}; }}")
            self.statusBar().showMessage("Focus Mode active. Press F11/Esc to exit.", 5000)
        else:
            self.text_edit.setStyleSheet("")
            # Reapply the correct theme stylesheet
            self.toggle_dark_mode(self.dark_mode_action.isChecked())

    def show_stats(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        text = self.text_edit.toPlainText(); dialog = DocumentStatsDialog(text, self); dialog.exec()
        QApplication.restoreOverrideCursor()
        
    def show_word_frequency(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        text = self.text_edit.toPlainText(); dialog = WordFrequencyDialog(text, self); dialog.exec()
        QApplication.restoreOverrideCursor()
        
    def show_readability(self) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        text = self.text_edit.toPlainText(); dialog = ReadabilityDialog(text, self); dialog.exec()
        QApplication.restoreOverrideCursor()

    def insert_chapter_break(self) -> None:
        self.text_edit.setFocus()
        if not self.text_edit.textCursor().atStart(): self.text_edit.insertPlainText("\f")
        self.text_edit.setAlignment(Qt.AlignmentFlag.AlignCenter); self.text_edit.setFontPointSize(16); self.text_edit.setFontWeight(QFont.Weight.Bold)
        doc_text = self.text_edit.toPlainText(); matches = re.findall(r'Chapter (\d+)', doc_text, re.IGNORECASE)
        next_num = 1
        if matches: next_num = max(int(n) for n in matches) + 1
        self.text_edit.insertPlainText(f"Chapter {next_num}")
        self.text_edit.setFontPointSize(12); self.text_edit.setFontWeight(QFont.Weight.Normal); self.text_edit.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.text_edit.insertPlainText("\n\n")

    def insert_scene_break(self) -> None:
        self.text_edit.setFocus()
        if not self.text_edit.textCursor().atBlockStart(): self.text_edit.insertPlainText("\n\n")
        self.text_edit.setAlignment(Qt.AlignmentFlag.AlignCenter); self.text_edit.insertPlainText("* * *"); self.text_edit.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.text_edit.insertPlainText("\n\n")

    def insert_name(self, genre: str) -> None:
        if genre == "Fantasy": name = random.choice(FANTASY_NAMES)
        else: name = random.choice(MODERN_NAMES)
        self.text_edit.insertPlainText(name)

    def format_screenplay(self, element_type: str) -> None:
        self.text_edit.setFocus(); cursor = self.text_edit.textCursor()
        block_fmt = QTextBlockFormat(); char_fmt = QTextCharFormat(self.text_edit.currentCharFormat()); char_fmt.setFontPointSize(12)
        if element_type == "Scene Heading":
            text = cursor.selectedText().upper() if cursor.hasSelection() else "INT. LOCATION - DAY"
            cursor.insertText(text); char_fmt.setFontWeight(QFont.Weight.Bold); block_fmt.setLeftMargin(0); block_fmt.setTopMargin(10); block_fmt.setBottomMargin(10)
        elif element_type == "Action":
            block_fmt.setLeftMargin(0); block_fmt.setTopMargin(10); block_fmt.setBottomMargin(10); char_fmt.setFontWeight(QFont.Weight.Normal)
        elif element_type == "Character":
            text = cursor.selectedText().upper() if cursor.hasSelection() else "CHARACTER NAME"
            cursor.insertText(text); block_fmt.setLeftMargin(180); block_fmt.setTopMargin(10)
        elif element_type == "Dialogue":
            block_fmt.setLeftMargin(180); block_fmt.setRightMargin(180); block_fmt.setTopMargin(2)
        elif element_type == "Parenthetical":
            text = cursor.selectedText() if cursor.hasSelection() else "beat"
            if not text.startswith("("): text = f"({text})"
            cursor.insertText(text); block_fmt.setLeftMargin(220); block_fmt.setRightMargin(150)
        elif element_type == "Transition":
            text = cursor.selectedText().upper() if cursor.hasSelection() else "CUT TO:"
            cursor.insertText(text); block_fmt.setAlignment(Qt.AlignmentFlag.AlignRight); block_fmt.setTopMargin(10)
        cursor.setBlockFormat(block_fmt); self.text_edit.setCurrentCharFormat(char_fmt)

    def on_text_changed(self) -> None:
        self.word_count_timer.start(500)  # Debounce word count calculation
        self.outline_timer.start(1000)

    def _calculate_word_count(self) -> None:
        text = self.text_edit.toPlainText()
        self.current_word_count = len(text.split())
        self.current_char_count = len(text)
        self.update_status_bar()

    def update_outline(self) -> None:
        text = self.text_edit.toPlainText()
        structure_key = re.findall(r'^\s*(Chapter\s+\d+|INT\.|EXT\.|\* \* \*)', text, re.MULTILINE)
        structure_key = tuple(structure_key)
        if hasattr(self, '_last_outline_key') and self._last_outline_key == structure_key:
            return
        self._last_outline_key = structure_key

        self.outline_list.clear(); doc = self.text_edit.document(); block = doc.begin()
        chapter_start_pos = 0; chapter_title = "Start"
        while block.isValid():
            text = block.text().strip()
            if text.startswith("Chapter"):
                word_count = self.get_word_count_in_range(chapter_start_pos, block.position())
                item = QListWidgetItem(f"{chapter_title} ({word_count} words)"); item.setData(Qt.ItemDataRole.UserRole, chapter_start_pos); self.outline_list.addItem(item)
                chapter_start_pos = block.position(); chapter_title = text
            elif text.startswith("INT.") or text.startswith("EXT."):
                item = QListWidgetItem(text); item.setData(Qt.ItemDataRole.UserRole, block.position()); item.setForeground(QColor(Qt.GlobalColor.darkBlue)); self.outline_list.addItem(item)
            elif text == "* * *":
                item = QListWidgetItem("Scene Break"); item.setData(Qt.ItemDataRole.UserRole, block.position()); self.outline_list.addItem(item)
            block = block.next()
        word_count = self.get_word_count_in_range(chapter_start_pos, doc.characterCount())
        item = QListWidgetItem(f"{chapter_title} ({word_count} words)"); item.setData(Qt.ItemDataRole.UserRole, chapter_start_pos); self.outline_list.addItem(item)

    def get_word_count_in_range(self, start: int, end: int) -> int:
        cursor = QTextCursor(self.text_edit.document()); cursor.setPosition(start); cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        text = cursor.selectedText(); return len(text.split())

    def navigate_to_outline_item(self, item) -> None:
        pos = item.data(Qt.ItemDataRole.UserRole); cursor = self.text_edit.textCursor(); cursor.setPosition(pos)
        self.text_edit.setTextCursor(cursor); self.text_edit.setFocus()

    def start_sprint(self) -> None:
        if self.sprint_active: self.stop_sprint(); return
        mins, ok = QInputDialog.getInt(self, "Writing Sprint", "Enter sprint duration (minutes):", 15, 1, 120)
        if not ok: return
        self.sprint_active = True; self.sprint_start_words = len(self.text_edit.toPlainText().split())
        self.sprint_seconds = mins * 60; self.sprint_timer.start(1000); self.sprint_btn.setText("Stop Sprint"); self.update_sprint_timer()

    def stop_sprint(self) -> None:
        self.sprint_timer.stop(); self.sprint_active = False; self.sprint_btn.setText("Start Sprint")
        current_words = len(self.text_edit.toPlainText().split()); words_written = current_words - self.sprint_start_words
        QMessageBox.information(self, "Sprint Complete!", f"You wrote {words_written} words!"); self.statusBar().setStyleSheet("")

    def update_sprint_timer(self) -> None:
        self.sprint_seconds -= 1
        if self.sprint_seconds <= 0: self.stop_sprint(); return
        mins, secs = divmod(self.sprint_seconds, 60); time_str = f"{mins:02d}:{secs:02d}"
        current_words = len(self.text_edit.toPlainText().split()); words_written = current_words - self.sprint_start_words
        self.statusBar().showMessage(f"Sprint: {time_str} | Words this sprint: {words_written}")
        self.statusBar().setStyleSheet("QStatusBar { background-color: #2c3e50; color: white; font-weight: bold; }")

    def toggle_dictation(self, checked: bool) -> None:
        if checked:
            self.dictation_thread = DictationThread()
            self.dictation_thread.text_recognized.connect(self.process_dictation); self.dictation_thread.error_occurred.connect(self.dictation_error)
            self.dictation_thread.start(); self.update_status_bar()
        else:
            if self.dictation_thread:
                self.dictation_thread.stop(); self.dictation_thread.wait(); self.dictation_thread = None
            self.update_status_bar()

    def process_dictation(self, text: str) -> None:
        # Preserves capitalization properly while applying punctuation replacements
        replacements = {
            r'\bperiod\b': '.', r'\bcomma\b': ',', r'\bquestion mark\b': '?', 
            r'\bexclamation mark\b': '!', r'\bexclamation point\b': '!',
            r'\bcolon\b': ':', r'\bsemicolon\b': ';', r'\bnew line\b': '\n', 
            r'\bnew paragraph\b': '\n\n', r'\bopen paren\b': '(', 
            r'\bclose paren\b': ')', r'\bopen quote\b': '“', r'\bclose quote\b': '”', 
            r'\bapostrophe s\b': "'s",
            r'\bdash\b': '—', r'\bellipsis\b': '…'
        }
        processed_text = text
        for pattern, repl in replacements.items():
            processed_text = re.sub(pattern, repl, processed_text, flags=re.IGNORECASE)
            
        processed_text = re.sub(r'\bquote\b', '“' if self.quote_toggle else '”', processed_text, flags=re.IGNORECASE)
        self.quote_toggle = not self.quote_toggle
        
        # Clean spaces before punctuation
        processed_text = re.sub(r'\s+([.,!?;:])', r'\1', processed_text)
        
        cursor = self.text_edit.textCursor()
        if cursor.atBlockStart():
            processed_text = processed_text[:1].upper() + processed_text[1:] if processed_text else processed_text
            
        # Intelligently append space only if text doesn't end with punctuation or newline
        if processed_text and not processed_text.endswith(('\n', ' ', '.', '!', '?', ',', ';', ':')):
            self.text_edit.insertPlainText(processed_text + " ")
        else:
            self.text_edit.insertPlainText(processed_text)
            
        self.text_edit.setTextCursor(cursor)

    def dictation_error(self, error_msg: str) -> None:
        QMessageBox.warning(self, "Dictation Error", error_msg); self.dictate_action.setChecked(False)

    def populate_voices(self) -> None:
        self.voices = self.tts.availableVoices(); self.voice_combo.clear()
        if self.voices:
            for voice in self.voices: self.voice_combo.addItem(voice.name())
            self.tts.setVoice(self.voices[0])
        else: self.voice_combo.addItem("No voices found"); self.voice_combo.setEnabled(False)

    def set_tts_voice(self, index: int) -> None:
        if self.voices and index < len(self.voices): self.tts.setVoice(self.voices[index])
    def set_tts_rate(self, value: int) -> None: self.tts.setRate(value / 100.0)
    def read_aloud(self) -> None:
        cursor = self.text_edit.textCursor()
        if cursor.hasSelection(): text = cursor.selectedText()
        else:
            cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor); text = cursor.selectedText()
            if not text: text = self.text_edit.toPlainText()
        if text: self.tts.say(text)
    def stop_reading(self) -> None: self.tts.stop()
    def pause_reading(self) -> None:
        if self.tts_state == QTextToSpeech.State.Ready: return
        elif self.tts_state == QTextToSpeech.State.Speaking: self.tts.pause(); self.pause_speak_action.setText("&Resume")
        elif self.tts_state == QTextToSpeech.State.Paused: self.tts.resume(); self.pause_speak_action.setText("&Pause")
    def on_tts_state_changed(self, state) -> None:
        self.tts_state = state
        if state == QTextToSpeech.State.Ready: self.pause_speak_action.setText("&Pause")
        self.update_status_bar()

    def zoom_in(self) -> None: self.zoom_level += 1; self.text_edit.zoomIn(1); self.update_zoom_label()
    def zoom_out(self) -> None: self.zoom_level -= 1; self.text_edit.zoomOut(1); self.update_zoom_label()
    def update_zoom_label(self) -> None: self.zoom_label.setText(f"{100 + (self.zoom_level * 10)}%")

    def toggle_dark_mode(self, checked: bool) -> None:
        if checked:
            self.setStyleSheet("""
                QMainWindow { background-color: #1e1e1e; }
                QToolBar { border: none; border-bottom: 1px solid #333; background-color: #252526; spacing: 2px; padding: 3px; }
                QToolBar QToolButton { padding: 4px; border-radius: 4px; background: transparent; color: #ccc; }
                QToolBar QToolButton:hover { background-color: #3a3d41; }
                QToolBar QToolButton:checked { background-color: #007acc; border: 1px solid #007acc; }
                QStatusBar { border-top: 1px solid #333; background-color: #007acc; color: white; font-weight: bold; }
                QStatusBar QLabel { border-left: 1px solid #13587e; color: #fff; }
                QStatusBar QPushButton { color: #fff; background: transparent; border: none; }
                QDockWidget { border: none; color: white; }
                QDockWidget::title { background-color: #252526; border-bottom: 1px solid #333; padding: 6px; font-weight: bold; }
                QListWidget { border: none; background-color: #252526; color: #ccc; }
                QTabWidget::pane { border: 1px solid #333; background-color: #1e1e1e; }
                QTabBar::tab { background: #2d2d2d; color: #ccc; border: 1px solid #333; padding: 6px 12px; margin-right: 2px; }
                QTabBar::tab:selected { background: #007acc; color: white; border-bottom: none; }
                QTextEdit { border: none; background-color: #1e1e1e; color: #d4d4d4; }
                QComboBox, QSpinBox, QLineEdit { border: 1px solid #3c3c3c; border-radius: 4px; padding: 2px 4px; background: #3c3c3c; color: white; }
                QPushButton { background-color: #0e639c; color: white; border: none; padding: 6px 12px; border-radius: 4px; }
                QPushButton:hover { background-color: #1177bb; }
                QMenu { background-color: #252526; border: 1px solid #333; color: white; }
                QMenu::item { padding: 6px 24px; border-radius: 4px; margin: 2px 4px; }
                QMenu::item:selected { background-color: #3a3d41; }
                QProgressBar { border: 1px solid #3c3c3c; border-radius: 4px; text-align: center; background: #252526; color: white; }
                QProgressBar::chunk { background-color: #0e639c; border-radius: 3px; }
                QScrollBar:vertical { border: none; background: #1e1e1e; width: 10px; margin: 0px; }
                QScrollBar::handle:vertical { background: #3c3c3c; min-height: 20px; border-radius: 4px; }
                QScrollBar::handle:vertical:hover { background: #4c4c4c; }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0px; }
                QScrollBar:horizontal { border: none; background: #1e1e1e; height: 10px; margin: 0px; }
                QScrollBar::handle:horizontal { background: #3c3c3c; min-width: 20px; border-radius: 4px; }
                QScrollBar::handle:horizontal:hover { background: #4c4c4c; }
                QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0px; }
            """)
        else:
            self.apply_modern_style()

    def change_dict_language(self, index: int) -> None:
        langs = ['en_US', 'en_GB', 'en_AU']
        if index < len(langs):
            self.text_edit.highlighter.set_language(langs[index])
            self.statusBar().showMessage(f"Dictionary changed to {langs[index]}", 3000)

    def toggle_academic_mode(self, checked: bool) -> None:
        self.text_edit.highlighter.set_academic_mode(checked)
        self.statusBar().showMessage(f"Academic Mode {'enabled' if checked else 'disabled'}", 3000)

    def eventFilter(self, obj, event) -> bool:
        if obj == self.text_edit.viewport() and event.type() == QEvent.Type.Wheel:
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                if event.angleDelta().y() > 0: self.zoom_in()
                else: self.zoom_out()
                return True
        return super().eventFilter(obj, event)

    def update_character_formatting(self) -> None:
        self.font_combo.blockSignals(True); self.font_size_combo.blockSignals(True); self.line_spacing_combo.blockSignals(True)
        fmt = self.text_edit.currentCharFormat(); self.font_combo.setCurrentFont(fmt.font())
        size = fmt.fontPointSize()
        if size > 0: self.font_size_combo.setCurrentText(str(int(size)))
        else: self.font_size_combo.setCurrentText("12")
        self.font_combo.blockSignals(False); self.font_size_combo.blockSignals(False)
        self.bold_action.setChecked(fmt.fontWeight() == QFont.Weight.Bold); self.italic_action.setChecked(fmt.fontItalic())
        self.underline_action.setChecked(fmt.fontUnderline()); self.strike_action.setChecked(fmt.fontStrikeOut())
        cursor = self.text_edit.textCursor(); self.bullet_list_action.setChecked(False); self.numbered_list_action.setChecked(False)
        if cursor.currentList():
            if cursor.currentList().format().style() == QTextListFormat.Style.ListDisc: self.bullet_list_action.setChecked(True)
            elif cursor.currentList().format().style() == QTextListFormat.Style.ListDecimal: self.numbered_list_action.setChecked(True)
        block_fmt = cursor.blockFormat()
        if block_fmt.lineHeightType() == QTextBlockFormat.LineHeightTypes.ProportionalHeight:
            factor = block_fmt.lineHeight() / 100.0; self.line_spacing_combo.setCurrentText(f"{factor:.1f}")
        else: self.line_spacing_combo.setCurrentText("1.0")
        alignment = self.text_edit.alignment()
        self.align_left_action.setChecked(alignment == Qt.AlignmentFlag.AlignLeft); self.align_center_action.setChecked(alignment == Qt.AlignmentFlag.AlignCenter)
        self.align_right_action.setChecked(alignment == Qt.AlignmentFlag.AlignRight); self.align_justify_action.setChecked(alignment == Qt.AlignmentFlag.AlignJustify)

    def update_cursor_position(self) -> None: self.update_status_bar()

    def update_status_bar(self) -> None:
        # Uses cached counts to prevent lag when moving cursor
        words = self.current_word_count
        chars = self.current_char_count
        cursor = self.text_edit.textCursor(); line = cursor.blockNumber() + 1; col = cursor.columnNumber() + 1
        if not self.sprint_active:
            status = f"Words: {words}  |  Characters: {chars}  |  Ln: {line}, Col: {col}"
            if self.dictation_thread: status = "🔴 Listening...  |  " + status
            elif self.tts_state == QTextToSpeech.State.Speaking: status = "🔊 Speaking...  |  " + status
            elif self.tts_state == QTextToSpeech.State.Paused: status = "⏸️ Paused...  |  " + status
            self.statusBar().showMessage(status)
        if self.word_goal > 0:
            self.goal_progress_bar.setValue(min(100, int((words / self.word_goal) * 100)) if self.word_goal > 0 else 0)
            if words >= self.word_goal: self.goal_progress_bar.setStyleSheet("QProgressBar::chunk { background-color: green; }")
            else: self.goal_progress_bar.setStyleSheet("")

    def set_word_goal(self) -> None:
        goal, ok = QInputDialog.getInt(self, "Set Word Count Goal", "Enter target word count:", self.word_goal, 0, 1000000, 100)
        if ok:
            self.word_goal = goal
            if goal > 0: self.goal_progress_bar.show(); self.goal_label.setText(f"Goal: {goal}")
            else: self.goal_progress_bar.hide(); self.goal_label.setText("Set Goal")
            self.update_status_bar()

    def closeEvent(self, event) -> None:
        if not self.maybe_save():
            event.ignore()
            return
        self.tts.stop()
        if self.dictation_thread:
            self.dictation_thread.stop(); self.dictation_thread.wait(); self.dictation_thread = None
        if self.text_edit.nlp_thread.isRunning():
            self.text_edit.nlp_thread.stop()
        if self.sprint_active: self.stop_sprint()
        self.story_bible_dock.save_notes()
        if not self.text_edit.document().isModified():
            if os.path.exists("recovery.html"): os.remove("recovery.html")
        self.save_settings(); event.accept()

# --- Supporting Dialogs ---
class DocumentStatsDialog(QDialog):
    """Dialog showing document word count, characters, and reading time."""
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Document Statistics")
        self.setFixedSize(300, 250)
        
        words = len(text.split())
        chars = len(text)
        paragraphs = len([b for b in text.split('\n') if b.strip() != ''])
        sentences = len(re.findall(r'[.!?]+', text))
        read_time = max(1, round(words / 200))
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Document Statistics</h2>"))
        
        form_box = QGroupBox("Text Metrics")
        form = QFormLayout(form_box)
        form.addRow("Words:", QLabel(f"<b>{words}</b>"))
        form.addRow("Characters:", QLabel(f"<b>{chars}</b>"))
        form.addRow("Paragraphs:", QLabel(f"<b>{paragraphs}</b>"))
        form.addRow("Sentences:", QLabel(f"<b>{sentences}</b>"))
        form.addRow("Avg Words/Sentence:", QLabel(f"<b>{round(words/sentences) if sentences else 0}</b>"))
        layout.addWidget(form_box)
        
        layout.addWidget(QLabel(f"⏱️ Estimated Reading Time: <b>{read_time} min</b>"))
        
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class ReadabilityDialog(QDialog):
    """Dialog showing Flesch reading ease and grade level."""
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Readability Analysis")
        self.setFixedSize(350, 300)
        
        words = re.findall(r'\b[A-Za-z]+\b', text)
        num_words = len(words)
        sentences = re.split(r'[.!?]+', text)
        num_sentences = max(1, len([s for s in sentences if s.strip()]))
        
        def count_syllables(word: str) -> int:
            word = word.lower()
            if len(word) <= 3: return 1
            word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
            word = re.sub(r'^y', '', word)
            matches = re.findall(r'[aeiouy]{1,2}', word)
            return max(1, len(matches))
            
        total_syllables = sum(count_syllables(w) for w in words)
        
        if num_words == 0:
            flesch_score = 0; grade_level = 0
        else:
            flesch_score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (total_syllables / num_words)
            grade_level = 0.39 * (num_words / num_sentences) + 11.8 * (total_syllables / num_words) - 15.59
            
        if flesch_score >= 90: interpretation = "Very easy to read (5th grade)"
        elif flesch_score >= 80: interpretation = "Easy to read (6th grade)"
        elif flesch_score >= 70: interpretation = "Fairly easy (7th grade)"
        elif flesch_score >= 60: interpretation = "Standard (8th-9th grade)"
        elif flesch_score >= 50: interpretation = "Fairly difficult (10th-12th)"
        elif flesch_score >= 30: interpretation = "Difficult (College level)"
        else: interpretation = "Very difficult (College Graduate)"
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Readability Analysis</h2>"))
        
        metrics_box = QGroupBox("Metrics")
        form = QFormLayout(metrics_box)
        form.addRow("Total Words:", QLabel(f"<b>{num_words}</b>"))
        form.addRow("Total Sentences:", QLabel(f"<b>{num_sentences}</b>"))
        form.addRow("Total Syllables:", QLabel(f"<b>{total_syllables}</b>"))
        form.addRow("Flesch Reading Ease:", QLabel(f"<b>{flesch_score:.2f}</b>"))
        form.addRow("Flesch-Kincaid Grade:", QLabel(f"<b>{grade_level:.2f}</b>"))
        layout.addWidget(metrics_box)
        
        layout.addWidget(QLabel(f"📊 Interpretation: <b>{interpretation}</b>"))
        
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class ProofreadingDialog(QDialog):
    """Dialog listing all spelling errors found in the document."""
    def __init__(self, editor, highlighter, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Proofreading Report")
        self.resize(500, 600)
        self.editor = editor
        self.highlighter = highlighter
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Proofreading Report</h2>"))
        
        self.summary_label = QLabel("Scanning...")
        layout.addWidget(self.summary_label)
        
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.go_to_error)
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        btn_layout.addWidget(btn)
        layout.addLayout(btn_layout)
        
        self.scan_document()

    def scan_document(self) -> None:
        self.list_widget.clear()
        doc = self.editor.document()
        block = doc.begin()
        count = 0
        while block.isValid():
            text = block.text()
            pos_offset = block.position()
            for match in re.finditer(r'\b[A-Za-z]+\b', text):
                word = match.group(0)
                if self.highlighter.is_misspelled(word):
                    if len(word) > 1:
                        item = QListWidgetItem(f"Line {block.blockNumber() + 1}: Spelling - '{word}'")
                        item.setData(Qt.ItemDataRole.UserRole, pos_offset + match.start())
                        self.list_widget.addItem(item)
                        count += 1
            block = block.next()
        self.summary_label.setText(f"Found {count} potential spelling errors.")

    def go_to_error(self, item) -> None:
        pos = item.data(Qt.ItemDataRole.UserRole)
        cursor = self.editor.textCursor()
        cursor.setPosition(pos)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()

class DictionaryManagerDialog(QDialog):
    """Dialog for adding/removing words from the custom dictionary."""
    def __init__(self, highlighter, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Custom Dictionary Manager")
        self.resize(400, 500)
        self.highlighter = highlighter
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Custom Dictionary</h2>"))
        layout.addWidget(QLabel("Words listed here will not be flagged as spelling errors."))
        
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.load_words()
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        delete_btn = QPushButton("Delete Selected")
        delete_btn.clicked.connect(self.delete_selected)
        btn_layout.addWidget(delete_btn)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def load_words(self) -> None:
        self.list_widget.clear()
        if os.path.exists("custom_dictionary.txt"):
            with open("custom_dictionary.txt", "r", encoding="utf-8") as f:
                words = sorted(f.read().splitlines())
                self.list_widget.addItems(words)

    def delete_selected(self) -> None:
        for item in self.list_widget.selectedItems():
            self.list_widget.takeItem(self.list_widget.row(item))
        self.save_dictionary()

    def save_dictionary(self) -> None:
        words = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
        with open("custom_dictionary.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(words))
        self.highlighter.ignored_words = set(words)
        self.highlighter.load_base_words()
        self.highlighter.rehighlight()

class WordFrequencyDialog(QDialog):
    """Dialog showing the most frequently used words."""
    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Word Frequency Analyzer")
        self.resize(400, 500)
        
        words = re.findall(r'\b[A-Za-z]+\b', text.lower())
        counter = Counter(words)
        
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Most Used Words</h2>"))
        layout.addWidget(QLabel("Use this to find and eliminate crutch words."))
        
        table = QTableWidget()
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Word", "Count"])
        table.setRowCount(len(counter))
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        
        for row, (word, count) in enumerate(counter.most_common()):
            table.setItem(row, 0, QTableWidgetItem(word))
            count_item = QTableWidgetItem(str(count))
            count_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(row, 1, count_item)
            
        layout.addWidget(table)
        
        btn = QPushButton("Close")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

class DictationThread(QThread):
    """Background thread for capturing microphone input and converting to text."""
    text_recognized = Signal(str)
    error_occurred = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.running = True

    def run(self) -> None:
        self.recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                while self.running:
                    try:
                        audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                        text = self.recognizer.recognize_google(audio)
                        self.text_recognized.emit(text)
                    except sr.WaitTimeoutError:
                        continue
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError as e:
                        self.error_occurred.emit(f"Network error: {e}")
                        break
        except OSError:
            self.error_occurred.emit("No microphone found or microphone is in use.")

    def stop(self) -> None:
        self.running = False

class FindReplaceDialog(QDialog):
    """Non-modal Find and Replace dialog."""
    def __init__(self, editor, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find and Replace")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowStaysOnTopHint)
        self.editor = editor
        self.setFixedSize(400, 180)

        layout = QVBoxLayout(self)

        find_layout = QHBoxLayout()
        find_layout.addWidget(QLabel("Find:"))
        self.find_input = QLineEdit()
        self.find_input.returnPressed.connect(self.find_next)
        find_layout.addWidget(self.find_input)
        self.find_btn = QPushButton("Find Next")
        self.find_btn.clicked.connect(self.find_next)
        find_layout.addWidget(self.find_btn)
        layout.addLayout(find_layout)

        replace_layout = QHBoxLayout()
        replace_layout.addWidget(QLabel("Replace:"))
        self.replace_input = QLineEdit()
        replace_layout.addWidget(self.replace_input)
        self.replace_btn = QPushButton("Replace")
        self.replace_btn.clicked.connect(self.replace)
        replace_layout.addWidget(self.replace_btn)
        layout.addLayout(replace_layout)

        btn_layout = QHBoxLayout()
        self.replace_all_btn = QPushButton("Replace All")
        self.replace_all_btn.clicked.connect(self.replace_all)
        btn_layout.addWidget(self.replace_all_btn)

        self.case_check = QPushButton("Match Case")
        self.case_check.setCheckable(True)
        btn_layout.addWidget(self.case_check)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)

    def find_next(self) -> None:
        text = self.find_input.text()
        if not text: return
        doc = self.editor.document()
        cursor = self.editor.textCursor()
        flags = QTextDocument.FindFlag.FindCaseSensitively if self.case_check.isChecked() else QTextDocument.FindFlag(0)
        find_cursor = doc.find(text, cursor, flags)
        if find_cursor.isNull():
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            find_cursor = doc.find(text, cursor, flags)
        if not find_cursor.isNull():
            self.editor.setTextCursor(find_cursor)

    def replace(self) -> None:
        text = self.find_input.text()
        repl = self.replace_input.text()
        cursor = self.editor.textCursor()
        if cursor.hasSelection() and cursor.selectedText() == text:
            cursor.insertText(repl)
        self.find_next()

    def replace_all(self) -> None:
        text = self.find_input.text()
        repl = self.replace_input.text()
        if not text: return
        
        if text == repl:
            QMessageBox.warning(self, "Replace All", "Search text and replacement text are identical.")
            return

        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        flags = QTextDocument.FindFlag.FindCaseSensitively if self.case_check.isChecked() else QTextDocument.FindFlag(0)
        count = 0
        
        while True:
            cursor = self.editor.document().find(text, cursor, flags)
            if cursor.isNull():
                break
            cursor.insertText(repl)
            count += 1
            
        cursor.endEditBlock()
        self.editor.document().setModified(True)
        QMessageBox.information(self, "Replace All", f"Replaced {count} occurrences.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    editor = WordProcessor()
    editor.show()
    sys.exit(app.exec())