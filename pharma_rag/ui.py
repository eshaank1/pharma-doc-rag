import gradio as gr

from .document_store import EnhancedDocumentStore

# ------------------------------------------------------------------
# Presentation helpers -- turn raw stats/strings into the styled HTML
# fragments used by the status card, the document structure list, and
# the status bar. Kept separate from the pipeline logic so the RAG code
# stays untouched; these only shape how results are displayed.
# ------------------------------------------------------------------

EMPTY_STATUS_HTML = """
<div class="status-card status-card--empty">
  <div class="status-empty-title">No document loaded</div>
  <div class="status-empty-sub">Upload a pharmaceutical blob PDF on the right to begin.</div>
</div>
"""


def format_status_card(stats):
    """Render processing stats as a certificate-style status card with
    a rotated 'VERIFIED' stamp, mirroring a real Certificate of Quality."""
    return f"""
<div class="status-card status-card--pass">
  <div class="status-stamp">VERIFIED</div>
  <div class="status-row"><span class="status-key">File</span><span class="status-val">{stats['filename']}</span></div>
  <div class="status-row"><span class="status-key">Pages</span><span class="status-val">{stats['total_pages']}</span></div>
  <div class="status-row"><span class="status-key">Documents found</span><span class="status-val">{stats['documents_found']}</span></div>
  <div class="status-row"><span class="status-key">Chunks created</span><span class="status-val">{stats['total_chunks']}</span></div>
  <div class="status-row status-row--wrap"><span class="status-key">Types</span><span class="status-val">{', '.join(stats['document_types'])}</span></div>
  <div class="status-row"><span class="status-key">Processing time</span><span class="status-val">{stats['processing_time']}</span></div>
</div>
"""


def format_status_error(message):
    return f"""
<div class="status-card status-card--error">
  <div class="status-empty-title">Processing failed</div>
  <div class="status-empty-sub">{message}</div>
</div>
"""


def format_doc_structure(structure):
    """Render the detected sub-documents as a list of chip rows instead
    of a plain bullet list."""
    if not structure:
        return ""
    rows = "".join(
        f"""<div class="doc-structure-item">
              <span class="doc-chip">{doc['type']}</span>
              <span class="doc-meta">Pages {doc['pages']} &middot; {doc['chunks']} chunks</span>
            </div>"""
        for doc in structure
    )
    return f'<div class="doc-structure">{rows}</div>'


def format_status_bar(stats=None):
    """Render the footer status bar as a row of pill-shaped stats
    instead of a single line of bold Markdown."""
    ready = stats is not None
    dot_class = "dot--pass" if ready else "dot"
    docs = stats.get('documents_found', 0) if stats else 0
    chunks = stats.get('total_chunks', 0) if stats else 0
    return f"""
<div class="statusbar">
  <span class="stat-pill"><span class="dot {dot_class}"></span>{'Ready' if ready else 'Idle'}</span>
  <span class="stat-pill">Documents <b>{docs}</b></span>
  <span class="stat-pill">Chunks <b>{chunks}</b></span>
</div>
"""


def format_sources_block(sources, confidence=None, filter_used=None):
    """Build a collapsible <details> block listing sources (and
    optionally confidence/filter), rendered as HTML inside the
    Markdown chat message so it starts collapsed and expands on click."""
    if not sources:
        return ""

    lines = "".join(
        f"<li>{src['doc_type']} (Pages {src['pages']})"
        + (f" - Relevance: {src['relevance']}" if 'relevance' in src else "")
        + "</li>"
        for src in sources
    )

    footer = ""
    if confidence is not None:
        footer = f'<div class="sources-footer"><em>Confidence: {confidence:.1%} | Filter: {filter_used}</em></div>'

    return (
        '<details class="sources-details">'
        '<summary>Sources</summary>'
        f'<ul>{lines}</ul>'
        f'{footer}'
        '</details>'
    )


# ------------------------------------------------------------------
# Theme -- a QC-lab / Certificate-of-Analysis palette rather than a
# generic chat-app theme: paper-white surfaces, deep clinical teal for
# affirmative actions, warm amber reserved for caution states, and a
# monospace face for anything that reads like lab data (filenames,
# lot numbers, page ranges, counts).
# ------------------------------------------------------------------

LAB_THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "SFMono-Regular", "monospace"],
).set(
    body_background_fill="#FAFAF7",
    background_fill_primary="#FFFFFF",
    background_fill_secondary="#F1F2ED",
    border_color_primary="#D9DDD5",
    block_background_fill="#FFFFFF",
    block_border_color="#D9DDD5",
    block_border_width="1px",
    block_radius="10px",
    block_label_text_color="#5B6660",
    block_label_text_weight="600",
    block_title_text_color="#1B2420",
    body_text_color="#1B2420",
    body_text_color_subdued="#5B6660",
    button_primary_background_fill="#0E6B57",
    button_primary_background_fill_hover="#0A4A3C",
    button_primary_text_color="#FFFFFF",
    button_primary_border_color="#0E6B57",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F2ED",
    button_secondary_border_color="#D9DDD5",
    button_secondary_text_color="#1B2420",
    input_background_fill="#FFFFFF",
    input_border_color="#D9DDD5",
    input_border_color_focus="#0E6B57",
    slider_color="#0E6B57",
    checkbox_background_color_selected="#0E6B57",
    checkbox_border_color_selected="#0E6B57",
    shadow_drop="0 1px 2px rgba(27,36,32,0.06)",
    # ---- Dark-mode variants pinned to the SAME light values. ----
    # Without these, components like File and Chatbot fall back to
    # Gradio's own built-in dark palette (and default font) whenever
    # the visitor's OS/browser is in dark mode. Pinning every *_dark
    # token stops the theme from switching at all.
    body_background_fill_dark="#FAFAF7",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary_dark="#F1F2ED",
    border_color_primary_dark="#D9DDD5",
    block_background_fill_dark="#FFFFFF",
    block_border_color_dark="#D9DDD5",
    block_label_text_color_dark="#5B6660",
    block_title_text_color_dark="#1B2420",
    body_text_color_dark="#1B2420",
    body_text_color_subdued_dark="#5B6660",
    button_primary_background_fill_dark="#0E6B57",
    button_primary_background_fill_hover_dark="#0A4A3C",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color_dark="#0E6B57",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover_dark="#F1F2ED",
    button_secondary_border_color_dark="#D9DDD5",
    button_secondary_text_color_dark="#1B2420",
    input_background_fill_dark="#FFFFFF",
    input_border_color_dark="#D9DDD5",
    input_border_color_focus_dark="#0E6B57",
    checkbox_background_color_selected_dark="#0E6B57",
    checkbox_border_color_selected_dark="#0E6B57",
)

COMPACT_CSS = """
:root {
    --paper: #FAFAF7;
    --paper-2: #F1F2ED;
    --ink: #1B2420;
    --ink-soft: #5B6660;
    --line: #D9DDD5;
    --teal: #0E6B57;
    --teal-dark: #0A4A3C;
    --teal-tint: #EAF5F1;
    --amber: #92621C;
    --amber-tint: #FBF1DF;
    --steel: #2F4B6E;
    --steel-tint: #EAF0F7;
}

.gradio-container { max-width: 100% !important; background: var(--paper) !important; }
#left_panel { font-size: 0.88em; }

/* Belt-and-suspenders: if the visitor's browser is in dark mode,
   Gradio scopes a `.dark` class onto the app root and reads its own
   CSS custom properties from it. Re-pin those to the same light
   values here so no component can silently switch palettes. */
.dark, .gradio-container.dark {
    --body-background-fill: var(--paper) !important;
    --background-fill-primary: #FFFFFF !important;
    --background-fill-secondary: var(--paper-2) !important;
    --border-color-primary: var(--line) !important;
    --block-background-fill: #FFFFFF !important;
    --block-border-color: var(--line) !important;
    --block-label-text-color: var(--ink-soft) !important;
    --block-title-text-color: var(--ink) !important;
    --body-text-color: var(--ink) !important;
    --body-text-color-subdued: var(--ink-soft) !important;
    --input-background-fill: #FFFFFF !important;
    --input-border-color: var(--line) !important;
    --button-secondary-background-fill: #FFFFFF !important;
    --button-secondary-border-color: var(--line) !important;
    --button-secondary-text-color: var(--ink) !important;
}

/* Force the type family everywhere, including inside components
   (File dropzone, Chatbot) whose internal markup sits below the
   selectors above and previously kept the browser's default font. */
.gradio-container, .gradio-container * {
    font-family: 'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif !important;
}
.status-val, .doc-chip, .doc-meta, .stat-pill, .hero-eyebrow,
.sources-details .sources-footer, code, pre {
    font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
}

/* -------------------- Hero / document header -------------------- */
.hero { padding: 2px 2px 16px 2px; border-bottom: 1px solid var(--line); margin-bottom: 16px; }
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72em; letter-spacing: 0.14em;
    text-transform: uppercase; color: var(--teal); margin-bottom: 8px;
}
.hero-title { font-size: 1.55em; font-weight: 650; margin: 0 0 6px 0; color: var(--ink); letter-spacing: -0.01em; }
.hero-sub { font-size: 0.92em; color: var(--ink-soft); margin: 0; max-width: 660px; line-height: 1.5; }

/* -------------------- Section labels (panel headers) -------------------- */
#left_panel h3, #right_panel h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72em !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: var(--ink-soft) !important;
    font-weight: 600 !important;
    border-bottom: 1px solid var(--line) !important;
    padding-bottom: 6px !important;
    margin: 14px 0 10px 0 !important;
}

.gr-button, button { padding: 4px 10px !important; }

/* Dropzone: compact height, wide horizontal layout, readable text size.
   Explicit background/text colors here (not just on .dark above)
   because the File component paints its own surface regardless of
   the light/dark class present on the root. */
#pdf_upload, #pdf_upload * {
    background: #FFFFFF !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}
#pdf_upload svg { color: var(--teal) !important; fill: currentColor !important; }
#pdf_upload { max-height: 130px !important; border-color: var(--line) !important; }
#pdf_upload .wrap {
    min-height: 90px !important; height: 90px !important;
    display: flex !important; flex-direction: row !important;
    align-items: center !important; justify-content: center !important;
    gap: 10px !important; flex-wrap: wrap !important;
}
#pdf_upload .wrap > * {
    display: flex !important; flex-direction: row !important;
    align-items: center !important; gap: 6px !important;
}
#pdf_upload .wrap, #pdf_upload .wrap * {
    font-size: 0.95em !important; line-height: 1.2 !important;
}
#pdf_upload .wrap svg { width: 20px !important; height: 20px !important; color: var(--teal) !important; }

/* Left panel: stretch to match right column without stray gaps
   between children (children default to flex-grow:1, which left
   empty space inside each block before a file is uploaded). Forcing
   flex: 0 0 auto packs everything from the top. */
#left_panel {
    display: flex !important; flex-direction: column !important;
    justify-content: flex-start !important; gap: 4px !important;
}
#left_panel > * { flex: 0 0 auto !important; }

/* Message textbox + Send button aligned in one row. */
#ask_row { display: flex !important; align-items: center !important; }
#ask_row > * { align-self: center !important; }

/* -------------------- Status card (Certificate-of-Analysis style) -------------------- */
.status-card {
    position: relative;
    background: var(--paper) !important;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px;
}
.status-card--pass {
    border-color: #BFE3D6;
    background: linear-gradient(180deg, var(--teal-tint), var(--paper) 65%) !important;
    padding-right: 84px;
}
.status-card--error { border-color: #E4C7AE; background: var(--amber-tint) !important; }
.status-card--empty { border-style: dashed; }
.status-empty-title { font-weight: 600; color: var(--ink); font-size: 0.92em; }
.status-empty-sub { color: var(--ink-soft); font-size: 0.85em; margin-top: 3px; }

.status-stamp {
    position: absolute; top: 10px; right: 12px;
    width: 60px; height: 60px; border-radius: 50%;
    border: 2px solid var(--teal); color: var(--teal);
    display: flex; align-items: center; justify-content: center; text-align: center;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.6em; font-weight: 700; letter-spacing: 0.03em;
    transform: rotate(-9deg); opacity: 0.9;
}
.status-stamp::after {
    content: ""; position: absolute; inset: 4px; border: 1px solid var(--teal); border-radius: 50%;
}

.status-row {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 14px; padding: 4px 0; font-size: 0.85em;
    border-bottom: 1px dashed var(--line);
}
.status-row:last-child { border-bottom: none; }
.status-key {
    color: var(--ink-soft);
    white-space: nowrap;
    flex-shrink: 0;
}
.status-val {
    font-family: 'IBM Plex Mono', monospace !important;
    color: var(--ink);
    text-align: right;
    word-break: break-word;
}
.status-row--wrap { align-items: flex-start; }
.status-row--wrap .status-val { text-align: left; }

/* -------------------- Document structure list -------------------- */
.doc-structure { display: flex; flex-direction: column; gap: 6px; margin-top: 8px; }
.doc-structure-item {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    font-size: 0.83em; padding: 6px 8px;
    background: var(--paper-2); border-radius: 8px; border: 1px solid var(--line);
}
.doc-chip {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.68em; padding: 2px 7px;
    border-radius: 999px; background: var(--steel-tint); color: var(--steel);
    white-space: nowrap; text-transform: uppercase; letter-spacing: 0.03em;
}
.doc-meta { color: var(--ink-soft); }

/* -------------------- Status bar (footer) -------------------- */
.statusbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 6px 2px; }
.stat-pill {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78em; color: var(--ink);
    background: var(--paper-2); border: 1px solid var(--line);
    padding: 4px 10px; border-radius: 999px; display: inline-flex; align-items: center; gap: 6px;
}
.stat-pill b { color: var(--teal); font-weight: 700; }
.dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ink-soft); display: inline-block; }
.dot--pass { background: var(--teal); }

/* -------------------- Chip-style utility buttons -------------------- */
.chip-btn, .chip-btn button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78em !important;
    border-radius: 999px !important;
    letter-spacing: 0.01em;
}

/* -------------------- Collapsible sources block inside chat -------------------- */
.sources-details {
    margin-top: 6px;
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 4px 8px;
    background: var(--paper-2);
}
.sources-details summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--teal-dark);
    padding: 4px 0;
    list-style: revert;
    font-size: 0.9em;
}
.sources-details summary:hover { opacity: 0.8; }
.sources-details ul { margin: 6px 0 2px 0; padding-left: 20px; }
.sources-details .sources-footer {
    margin-top: 4px;
    opacity: 0.75;
    font-size: 0.85em;
    font-family: 'IBM Plex Mono', monospace;
}

/* -------------------- Chatbot -------------------- */
/* Same reasoning as #pdf_upload above: force every descendant, not
   just the outer frame, so the empty-state canvas and message
   bubbles can't fall back to Gradio's dark surface. */
#chatbot, #chatbot * {
    background: var(--paper) !important;
    color: var(--ink) !important;
    border-color: var(--line) !important;
}
#chatbot { border: 1px solid var(--line) !important; }
#chatbot .message.user, #chatbot [data-testid="user"] {
    background: var(--teal-tint) !important;
}
#chatbot .message.bot, #chatbot [data-testid="bot"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
}
"""


def create_interface():
    """Create the Gradio interface for pharmaceutical document Q&A."""

    doc_store = EnhancedDocumentStore()

    def process_pdf_handler(pdf_file):
        """Handle PDF upload and processing."""
        if pdf_file is None:
            return EMPTY_STATUS_HTML, "", gr.update(choices=["All"])

        success, stats = doc_store.process_pdf(
            pdf_file,
            filename=pdf_file.split('/')[-1] if isinstance(pdf_file, str) else
            getattr(pdf_file, 'name', 'pharma-blob-sample.pdf')
        )

        if success:
            status_msg = format_status_card(stats)
            structure = doc_store.get_document_structure()
            structure_display = format_doc_structure(structure)
            doc_types = ["All"] + stats['document_types']
            return status_msg, structure_display, gr.update(choices=doc_types, value="All")
        else:
            return format_status_error(stats.get('error', 'Unknown error')), "", gr.update(choices=["All"])

    def chat_handler(message, history, doc_filter, auto_route, num_chunks):
        """Handle chat interactions."""
        if not doc_store.is_ready:
            response = "Please upload and process a pharmaceutical PDF document first."
            return history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

        filter_type = None if doc_filter == "All" else doc_filter
        result = doc_store.query(
            message,
            filter_type=filter_type,
            auto_route=auto_route and filter_type is None,
            k=num_chunks
        )

        response = f"{result['answer']}\n\n"
        response += format_sources_block(result['sources'], result['confidence'], result['filter_used'])

        return history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]

    with gr.Blocks() as demo:
        gr.HTML("""
        <div class="hero">
            <div class="hero-eyebrow">RAG-Assisted Review &middot; Local Inference Only</div>
            <h1 class="hero-title">Pharmaceutical Document Q&amp;A</h1>
            <p class="hero-sub">Upload a pharmaceutical blob PDF (e.g. pharma-blob-sample.pdf) to identify
            document types, build a searchable index, and ask questions in natural language.</p>
        </div>
        """)

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_id="left_panel"):
                gr.Markdown("### Document Info")
                status_output = gr.Markdown(value=EMPTY_STATUS_HTML)
                structure_output = gr.Markdown(value="", label="Document Structure")

                gr.Markdown("### Retrieval Settings")

                doc_filter = gr.Dropdown(
                    choices=["All"],
                    value="All",
                    label="Document Type Filter",
                    info="Filter search to a specific pharmaceutical document type"
                )

                auto_route = gr.Checkbox(
                    value=True,
                    label="Auto-Route Queries",
                    info="Automatically detect the most relevant document type"
                )

                num_chunks = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=4,
                    step=1,
                    label="Chunks to Retrieve"
                )

            with gr.Column(scale=4, elem_id="right_panel"):
                with gr.Group():
                    pdf_input = gr.File(
                        label="Upload Pharmaceutical PDF",
                        file_types=[".pdf"],
                        type="filepath",
                        elem_id="pdf_upload"
                    )

                    with gr.Row():
                        process_btn = gr.Button(
                            "Process Document",
                            variant="primary",
                            size="sm",
                            scale=2
                        )
                        clear_all_btn = gr.Button(
                            "Clear All",
                            variant="secondary",
                            size="sm",
                            scale=1
                        )

                gr.Markdown("### Ask Questions")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=560,
                    elem_id="chatbot",
                    show_label=False,
                    buttons=["copy", "copy_all"],  # omit "share": it only
                    # works on a real HF Space (posts to a Spaces
                    # Discussion thread); it has nowhere to send the
                    # conversation when self-hosted, so the click would
                    # silently do nothing.
                )

                with gr.Row(elem_id="ask_row"):
                    msg_input = gr.Textbox(
                        label="Ask a question",
                        placeholder="e.g., What is the lot number? What sterilization method was used?",
                        scale=4,
                        show_label=False
                    )
                    send_btn = gr.Button("Send", scale=1, variant="primary")

                with gr.Row():
                    clear_chat_btn = gr.Button("Clear Chat", size="sm", scale=1, elem_classes=["chip-btn"])
                    example_btn1 = gr.Button("Summarize Document", size="sm", scale=1, elem_classes=["chip-btn"])
                    example_btn2 = gr.Button("Find Lot Numbers", size="sm", scale=1, elem_classes=["chip-btn"])

        with gr.Row():
            status_bar = gr.Markdown(value=format_status_bar(), elem_id="status_bar")

        def update_status_bar():
            """Update the status bar with current statistics."""
            if doc_store.is_ready:
                return format_status_bar(doc_store.processing_stats)
            return format_status_bar()

        def clear_all():
            """Clear everything and reset the interface."""
            nonlocal doc_store
            doc_store = EnhancedDocumentStore()
            return (
                None,  # pdf_input
                EMPTY_STATUS_HTML,  # status_output
                "",  # structure_output
                gr.update(choices=["All"], value="All"),  # doc_filter
                [],  # chatbot
                "",  # msg_input
            )

        def ask_summary(history):
            result = doc_store.summarize_all_documents()
            response = f"{result['answer']}\n\n"
            response += format_sources_block(result['sources'])
            message = "Can you provide a summary of the main points in this document?"
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response}
            ]

        def ask_lot_numbers(history):
            return chat_handler(
                "What lot numbers or batch numbers are mentioned in these documents?",
                history, doc_filter.value, auto_route.value, num_chunks.value
            )

        # Wire up all the events.
        #
        # LOADING-BOX FIX: Gradio shows a pending overlay (spinner +
        # "x.x/xx.xs" eta text) on every OUTPUT component of a running
        # event. status_bar is a thin, mostly-empty row pinned to the
        # bottom of the page, so bundling it into the same event as
        # chatbot/status_output made that overlay render as its own
        # floating box down there instead of staying inside the chat
        # panel. Fix: update status_bar in a separate chained step with
        # show_progress="hidden", so only the chatbot (or the status
        # card, for PDF processing) shows a loading indicator.
        process_btn.click(
            fn=process_pdf_handler,
            inputs=[pdf_input],
            outputs=[status_output, structure_output, doc_filter]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        )

        clear_all_btn.click(
            fn=clear_all,
            outputs=[pdf_input, status_output, structure_output, doc_filter,
                     chatbot, msg_input]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        )

        msg_input.submit(
            fn=chat_handler,
            inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        ).then(
            lambda: "",
            outputs=[msg_input],
            show_progress="hidden"
        )

        send_btn.click(
            fn=chat_handler,
            inputs=[msg_input, chatbot, doc_filter, auto_route, num_chunks],
            outputs=[chatbot]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        ).then(
            lambda: "",
            outputs=[msg_input],
            show_progress="hidden"
        )

        clear_chat_btn.click(
            lambda: [],
            outputs=[chatbot]
        )

        example_btn1.click(
            fn=ask_summary,
            inputs=[chatbot],
            outputs=[chatbot]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        )

        example_btn2.click(
            fn=ask_lot_numbers,
            inputs=[chatbot],
            outputs=[chatbot]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        )

        pdf_input.change(
            fn=process_pdf_handler,
            inputs=[pdf_input],
            outputs=[status_output, structure_output, doc_filter]
        ).then(
            fn=update_status_bar,
            outputs=[status_bar],
            show_progress="hidden"
        )

    return demo
