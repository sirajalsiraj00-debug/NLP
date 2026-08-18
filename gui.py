import os
import shutil
import customtkinter as ctk

from tkinter import filedialog, messagebox

from main import (
    load_documents,
    build_vectorizer,
    search_by_text,
    search_by_document
)

# =========================================================
# THEME
# =========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BG = "#080B12"
SIDEBAR = "#0D111A"
CARD = "#111722"
CARD_HOVER = "#171F2D"
BORDER = "#202A3A"

TEXT = "#F3F5F7"
TEXT_SECONDARY = "#8E9AAF"
TEXT_MUTED = "#5E6A7D"

BLUE = "#4F8CFF"
BLUE_HOVER = "#6A9DFF"

PURPLE = "#8B5CF6"
GOLD = "#D6B36A"

GREEN = "#35D07F"
RED = "#EF6262"

# =========================================================
# APPLICATION
# =========================================================

class SimilarityApp(ctk.CTk):

    def __init__(self):

        super().__init__()

        self.title(
            "Docusense AI — Document Similarity Engine"
        )

        self.geometry(
            "1450x900"
        )

        self.minsize(
            1200,
            750
        )

        self.configure(
            fg_color=BG
        )

        # -------------------------------------------------
        # Data
        # -------------------------------------------------

        self.documents = []
        self.document_names = []
        self.raw_documents = []

        self.vectorizer = None
        self.tfidf_matrix = None

        self.selected_query_pdf = None
        self.search_mode = "Text"

        # -------------------------------------------------
        # UI
        # -------------------------------------------------

        self.create_sidebar()
        self.create_main_area()

        self.load_corpus()

    # =====================================================
    # SIDEBAR
    # =====================================================

    def create_sidebar(self):

        self.sidebar = ctk.CTkFrame(
            self,
            width=285,
            corner_radius=0,
            fg_color=SIDEBAR
        )

        self.sidebar.pack(
            side="left",
            fill="y"
        )

        self.sidebar.pack_propagate(
            False
        )

        # -------------------------------------------------
        # Brand
        # -------------------------------------------------

        brand_frame = ctk.CTkFrame(
            self.sidebar,
            fg_color="transparent"
        )

        brand_frame.pack(
            padx=25,
            pady=(35, 35),
            fill="x"
        )

        logo = ctk.CTkLabel(
            brand_frame,
            text="◈",
            text_color=BLUE,
            font=ctk.CTkFont(
                size=34,
                weight="bold"
            )
        )

        logo.pack(
            side="left",
            padx=(0, 10)
        )

        brand_text = ctk.CTkFrame(
            brand_frame,
            fg_color="transparent"
        )

        brand_text.pack(
            side="left"
        )

        title = ctk.CTkLabel(
            brand_text,
            text="DOCUSENSE",
            text_color=TEXT,
            font=ctk.CTkFont(
                size=18,
                weight="bold"
            )
        )

        title.pack(
            anchor="w"
        )

        subtitle = ctk.CTkLabel(
            brand_text,
            text="AI DOCUMENT ENGINE",
            text_color=GOLD,
            font=ctk.CTkFont(
                size=9,
                weight="bold"
            )
        )

        subtitle.pack(
            anchor="w"
        )

        # -------------------------------------------------
        # Navigation
        # -------------------------------------------------

        self.section_label(
            "WORKSPACE"
        )

        self.nav_button(
            "⌕   Similarity Search",
            active=True
        )

        self.nav_button(
            "▣   Document Corpus"
        )

        self.nav_button(
            "◉   Analytics"
        )

        self.nav_button(
            "⚙   Engine Settings"
        )

        # -------------------------------------------------
        # Corpus
        # -------------------------------------------------

        self.section_label(
            "DOCUMENT CORPUS"
        )

        self.corpus_list = ctk.CTkScrollableFrame(
            self.sidebar,
            fg_color="transparent",
            height=250
        )

        self.corpus_list.pack(
            padx=15,
            pady=(0, 10),
            fill="both",
            expand=True
        )

        # -------------------------------------------------
        # Add Documents
        # -------------------------------------------------

        self.add_button = ctk.CTkButton(
            self.sidebar,
            text="+  ADD DOCUMENTS",
            height=42,
            corner_radius=10,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            command=self.add_documents
        )

        self.add_button.pack(
            padx=20,
            pady=(10, 8),
            fill="x"
        )

        self.rebuild_button = ctk.CTkButton(
            self.sidebar,
            text="↻  REBUILD INDEX",
            height=38,
            corner_radius=10,
            fg_color=CARD,
            hover_color=CARD_HOVER,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT_SECONDARY,
            command=self.rebuild_index
        )

        self.rebuild_button.pack(
            padx=20,
            pady=(0, 20),
            fill="x"
        )

    # =====================================================
    # SIDEBAR HELPERS
    # =====================================================

    def section_label(self, text):

        label = ctk.CTkLabel(
            self.sidebar,
            text=text,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                size=10,
                weight="bold"
            )
        )

        label.pack(
            padx=25,
            pady=(5, 10),
            anchor="w"
        )

    def nav_button(
        self,
        text,
        active=False
    ):

        button = ctk.CTkButton(
            self.sidebar,
            text=text,
            height=42,
            corner_radius=9,
            anchor="w",
            fg_color=(
                "#182238"
                if active
                else "transparent"
            ),
            hover_color=CARD_HOVER,
            text_color=(
                TEXT
                if active
                else TEXT_SECONDARY
            ),
            font=ctk.CTkFont(
                size=12,
                weight="bold"
                if active
                else "normal"
            )
        )

        button.pack(
            padx=15,
            pady=2,
            fill="x"
        )

    # =====================================================
    # MAIN AREA
    # =====================================================

    def create_main_area(self):

        self.main = ctk.CTkFrame(
            self,
            corner_radius=0,
            fg_color=BG
        )

        self.main.pack(
            side="right",
            fill="both",
            expand=True
        )

        # -------------------------------------------------
        # Header
        # -------------------------------------------------

        header = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        header.pack(
            padx=40,
            pady=(35, 20),
            fill="x"
        )

        header_left = ctk.CTkFrame(
            header,
            fg_color="transparent"
        )

        header_left.pack(
            side="left"
        )

        ctk.CTkLabel(
            header_left,
            text="Document Similarity",
            text_color=TEXT,
            font=ctk.CTkFont(
                size=31,
                weight="bold"
            )
        ).pack(
            anchor="w"
        )

        ctk.CTkLabel(
            header_left,
            text=(
                "Intelligent document analysis "
                "powered by TF-IDF & Cosine Similarity"
            ),
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(
                size=13
            )
        ).pack(
            anchor="w",
            pady=(5, 0)
        )

        # Status

        status = ctk.CTkFrame(
            header,
            fg_color=CARD,
            corner_radius=20,
            border_width=1,
            border_color=BORDER
        )

        status.pack(
            side="right",
            padx=5,
            pady=5
        )

        ctk.CTkLabel(
            status,
            text="●",
            text_color=GREEN,
            font=ctk.CTkFont(
                size=12
            )
        ).pack(
            side="left",
            padx=(12, 5)
        )

        self.status_label = ctk.CTkLabel(
            status,
            text="ENGINE READY",
            text_color=TEXT_SECONDARY,
            font=ctk.CTkFont(
                size=10,
                weight="bold"
            )
        )

        self.status_label.pack(
            side="left",
            padx=(0, 12),
            pady=8
        )

        # -------------------------------------------------
        # Statistics
        # -------------------------------------------------

        stats = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        stats.pack(
            padx=40,
            pady=(0, 20),
            fill="x"
        )

        self.documents_card = self.stat_card(
            stats,
            "DOCUMENTS",
            "0",
            BLUE
        )

        self.documents_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(0, 8)
        )

        self.features_card = self.stat_card(
            stats,
            "TF-IDF FEATURES",
            "0",
            PURPLE
        )

        self.features_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=8
        )

        self.index_card = self.stat_card(
            stats,
            "INDEX STATUS",
            "READY",
            GOLD
        )

        self.index_card.pack(
            side="left",
            fill="both",
            expand=True,
            padx=(8, 0)
        )

        # -------------------------------------------------
        # Search Card
        # -------------------------------------------------

        search_card = ctk.CTkFrame(
            self.main,
            fg_color=CARD,
            corner_radius=16,
            border_width=1,
            border_color=BORDER
        )

        search_card.pack(
            padx=40,
            pady=(0, 20),
            fill="x"
        )

        search_header = ctk.CTkFrame(
            search_card,
            fg_color="transparent"
        )

        search_header.pack(
            padx=22,
            pady=(20, 10),
            fill="x"
        )

        ctk.CTkLabel(
            search_header,
            text="SEARCH QUERY",
            text_color=TEXT,
            font=ctk.CTkFont(
                size=14,
                weight="bold"
            )
        ).pack(
            side="left"
        )

        self.mode_label = ctk.CTkLabel(
            search_header,
            text="TEXT MODE",
            text_color=BLUE,
            font=ctk.CTkFont(
                size=10,
                weight="bold"
            )
        )

        self.mode_label.pack(
            side="right"
        )

        # Query

        self.query_text = ctk.CTkTextbox(
            search_card,
            height=105,
            corner_radius=10,
            fg_color="#0B1018",
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(
                size=13
            )
        )

        self.query_text.pack(
            padx=22,
            pady=5,
            fill="x"
        )

        self.query_text.insert(
            "1.0",
            "Enter a sentence or document-related query..."
        )

        # Controls

        controls = ctk.CTkFrame(
            search_card,
            fg_color="transparent"
        )

        controls.pack(
            padx=22,
            pady=(10, 20),
            fill="x"
        )

        self.text_mode_button = ctk.CTkButton(
            controls,
            text="TEXT",
            width=100,
            height=36,
            corner_radius=8,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            command=lambda:
            self.set_search_mode("Text")
        )

        self.text_mode_button.pack(
            side="left",
            padx=(0, 7)
        )

        self.pdf_mode_button = ctk.CTkButton(
            controls,
            text="PDF",
            width=100,
            height=36,
            corner_radius=8,
            fg_color="#1A2230",
            hover_color=CARD_HOVER,
            command=lambda:
            self.set_search_mode("PDF")
        )

        self.pdf_mode_button.pack(
            side="left"
        )

        self.pdf_button = ctk.CTkButton(
            controls,
            text="SELECT PDF",
            width=150,
            height=36,
            corner_radius=8,
            fg_color="#1A2230",
            hover_color=CARD_HOVER,
            command=self.select_query_pdf
        )

        self.search_button = ctk.CTkButton(
            controls,
            text="SEARCH  →",
            width=150,
            height=40,
            corner_radius=9,
            fg_color=BLUE,
            hover_color=BLUE_HOVER,
            font=ctk.CTkFont(
                size=12,
                weight="bold"
            ),
            command=self.perform_search
        )

        self.search_button.pack(
            side="right"
        )

        # -------------------------------------------------
        # Results Header
        # -------------------------------------------------

        results_header = ctk.CTkFrame(
            self.main,
            fg_color="transparent"
        )

        results_header.pack(
            padx=40,
            pady=(0, 10),
            fill="x"
        )

        ctk.CTkLabel(
            results_header,
            text="SIMILAR DOCUMENTS",
            text_color=TEXT,
            font=ctk.CTkFont(
                size=15,
                weight="bold"
            )
        ).pack(
            side="left"
        )

        self.result_count_label = ctk.CTkLabel(
            results_header,
            text="TOP 3",
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                size=10,
                weight="bold"
            )
        )

        self.result_count_label.pack(
            side="right"
        )

        # -------------------------------------------------
        # Results
        # -------------------------------------------------

        self.results_box = ctk.CTkTextbox(
            self.main,
            corner_radius=14,
            fg_color=CARD,
            border_width=1,
            border_color=BORDER,
            text_color=TEXT,
            font=ctk.CTkFont(
                size=13
            )
        )

        self.results_box.pack(
            padx=40,
            pady=(0, 35),
            fill="both",
            expand=True
        )

    # =====================================================
    # STAT CARD
    # =====================================================

    def stat_card(
        self,
        parent,
        title,
        value,
        accent
    ):

        card = ctk.CTkFrame(
            parent,
            height=95,
            fg_color=CARD,
            corner_radius=14,
            border_width=1,
            border_color=BORDER
        )

        card.pack_propagate(
            False
        )

        ctk.CTkLabel(
            card,
            text=title,
            text_color=TEXT_MUTED,
            font=ctk.CTkFont(
                size=9,
                weight="bold"
            )
        ).pack(
            padx=18,
            pady=(15, 0),
            anchor="w"
        )

        value_label = ctk.CTkLabel(
            card,
            text=value,
            text_color=accent,
            font=ctk.CTkFont(
                size=24,
                weight="bold"
            )
        )

        value_label.pack(
            padx=18,
            pady=(2, 10),
            anchor="w"
        )

        card.value_label = value_label

        return card

    # =====================================================
    # LOAD CORPUS
    # =====================================================

    def load_corpus(self):

        folder = "documents"

        (
            self.documents,
            self.document_names,
            self.raw_documents
        ) = load_documents(
            folder
        )

        if self.documents:

            (
                self.vectorizer,
                self.tfidf_matrix
            ) = build_vectorizer(
                self.documents
            )

            self.status_label.configure(
                text="ENGINE READY"
            )

        else:

            self.vectorizer = None
            self.tfidf_matrix = None

            self.status_label.configure(
                text="NO DOCUMENTS"
            )

        self.update_statistics()
        self.update_corpus_list()

    # =====================================================
    # UPDATE STATISTICS
    # =====================================================

    def update_statistics(self):

        self.documents_card.value_label.configure(
            text=str(
                len(self.documents)
            )
        )

        if self.vectorizer:

            features = len(
                self.vectorizer
                .get_feature_names_out()
            )

        else:

            features = 0

        self.features_card.value_label.configure(
            text=f"{features:,}"
        )

        self.index_card.value_label.configure(
            text=(
                "READY"
                if self.vectorizer
                else "EMPTY"
            )
        )

    # =====================================================
    # CORPUS LIST
    # =====================================================

    def update_corpus_list(self):

        for widget in (
            self.corpus_list.winfo_children()
        ):

            widget.destroy()

        for index, name in enumerate(
            self.document_names,
            start=1
        ):

            row = ctk.CTkFrame(
                self.corpus_list,
                height=35,
                corner_radius=7,
                fg_color="transparent"
            )

            row.pack(
                padx=2,
                pady=2,
                fill="x"
            )

            ctk.CTkLabel(
                row,
                text=f"{index:02d}",
                text_color=TEXT_MUTED,
                width=25,
                font=ctk.CTkFont(
                    size=9
                )
            ).pack(
                side="left"
            )

            ctk.CTkLabel(
                row,
                text=name,
                text_color=TEXT_SECONDARY,
                anchor="w",
                font=ctk.CTkFont(
                    size=10
                )
            ).pack(
                side="left",
                fill="x",
                expand=True
            )

    # =====================================================
    # ADD DOCUMENTS
    # =====================================================

    def add_documents(self):

        files = filedialog.askopenfilenames(
            title="Select PDF Documents",
            filetypes=[
                (
                    "PDF Documents",
                    "*.pdf"
                )
            ]
        )

        if not files:
            return

        os.makedirs(
            "documents",
            exist_ok=True
        )

        added = 0

        for file_path in files:

            filename = os.path.basename(
                file_path
            )

            destination = os.path.join(
                "documents",
                filename
            )

            if os.path.abspath(
                file_path
            ) == os.path.abspath(
                destination
            ):

                continue

            shutil.copy2(
                file_path,
                destination
            )

            added += 1

        self.load_corpus()

        messagebox.showinfo(
            "Corpus Updated",
            f"{added} document(s) added successfully."
        )

    # =====================================================
    # REBUILD INDEX
    # =====================================================

    def rebuild_index(self):

        self.status_label.configure(
            text="BUILDING INDEX..."
        )

        self.update()

        self.load_corpus()

        messagebox.showinfo(
            "Index Rebuilt",
            "TF-IDF index rebuilt successfully."
        )

    # =====================================================
    # SEARCH MODE
    # =====================================================

    def set_search_mode(
        self,
        mode
    ):

        self.search_mode = mode

        if mode == "Text":

            self.mode_label.configure(
                text="TEXT MODE",
                text_color=BLUE
            )

            self.text_mode_button.configure(
                fg_color=BLUE
            )

            self.pdf_mode_button.configure(
                fg_color="#1A2230"
            )

            self.pdf_button.pack_forget()

            self.query_text.configure(
                state="normal"
            )

        else:

            self.mode_label.configure(
                text="PDF MODE",
                text_color=PURPLE
            )

            self.text_mode_button.configure(
                fg_color="#1A2230"
            )

            self.pdf_mode_button.configure(
                fg_color=PURPLE
            )

            self.pdf_button.pack(
                side="left",
                padx=(12, 0)
            )

            self.query_text.configure(
                state="disabled"
            )

    # =====================================================
    # SELECT QUERY PDF
    # =====================================================

    def select_query_pdf(self):

        file_path = filedialog.askopenfilename(
            title="Select Query PDF",
            filetypes=[
                (
                    "PDF Documents",
                    "*.pdf"
                )
            ]
        )

        if not file_path:
            return

        self.selected_query_pdf = file_path

        self.pdf_button.configure(
            text=os.path.basename(
                file_path
            )
        )

    # =====================================================
    # SEARCH
    # =====================================================

    def perform_search(self):

        if not self.documents:

            messagebox.showwarning(
                "No Corpus",
                "Add PDF documents before searching."
            )

            return

        self.results_box.delete(
            "1.0",
            "end"
        )

        self.status_label.configure(
            text="ANALYZING..."
        )

        self.update()

        try:

            if self.search_mode == "Text":

                query = self.query_text.get(
                    "1.0",
                    "end"
                ).strip()

                if (
                    not query
                    or
                    query.startswith(
                        "Enter a sentence"
                    )
                ):

                    messagebox.showwarning(
                        "Empty Query",
                        "Enter a search query."
                    )

                    self.status_label.configure(
                        text="ENGINE READY"
                    )

                    return

                results = search_by_text(
                    query,
                    self.vectorizer,
                    self.tfidf_matrix,
                    self.document_names,
                    self.raw_documents,
                    top_k=3
                )

            else:

                if not self.selected_query_pdf:

                    messagebox.showwarning(
                        "No PDF",
                        "Select a PDF query first."
                    )

                    self.status_label.configure(
                        text="ENGINE READY"
                    )

                    return

                results = search_by_document(
                    self.selected_query_pdf,
                    self.vectorizer,
                    self.tfidf_matrix,
                    self.document_names,
                    self.raw_documents,
                    top_k=3
                )

            self.display_results(
                results
            )

            self.status_label.configure(
                text="ANALYSIS COMPLETE"
            )

        except Exception as error:

            self.status_label.configure(
                text="ENGINE ERROR"
            )

            messagebox.showerror(
                "Search Error",
                str(error)
            )

    # =====================================================
    # DISPLAY RESULTS
    # =====================================================

    def display_results(
        self,
        results
    ):

        if not results:

            self.results_box.insert(
                "end",
                "\n  No similar documents found."
            )

            return

        for rank, result in enumerate(
            results,
            start=1
        ):

            score = (
                result["score"] * 100
            )

            self.results_box.insert(
                "end",
                "\n"
                f"  #{rank}   "
                f"{result['document']}\n"
            )

            self.results_box.insert(
                "end",
                f"  Similarity        "
                f"{score:.2f}%\n"
            )

            # Progress bar representation

            blocks = int(
                score / 5
            )

            bar = (
                "█" * blocks
                +
                "░" * (20 - blocks)
            )

            self.results_box.insert(
                "end",
                f"  {bar}\n"
            )

            self.results_box.insert(
                "end",
                f"  Shared Features   "
                f"{result['common_count']}\n"
            )

            if result["common_terms"]:

                terms = ", ".join(
                    result[
                        "common_terms"
                    ][:12]
                )

                self.results_box.insert(
                    "end",
                    f"  Keywords           "
                    f"{terms}\n"
                )

            self.results_box.insert(
                "end",
                "\n  MOST SIMILAR SENTENCES\n"
            )

            for sentence_rank, sentence in enumerate(
                result.get("similar_sentences",[]),
                start=1
            ):

                sentence_score = (
                    sentence["score"] * 100
                )

                self.results_box.insert(
                    "end",
                    f"\n  {sentence_rank}. "
                    f"{sentence_score:.2f}%\n"
                )

                self.results_box.insert(
                    "end",
                    f"     "
                    f"{sentence['sentence']}\n"
                )

            self.results_box.insert(
                "end",
                "\n"
                + "─" * 90
                + "\n"
            )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app = SimilarityApp()

    app.mainloop()