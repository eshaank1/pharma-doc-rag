from pathlib import Path

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

COMPACT_CSS = Path(__file__).with_name("style.css").read_text()


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
                    buttons=["copy", "copy_all"],  # omit "share"
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
        # status_bar is updated in a separate .then() with
        # show_progress="hidden" -- bundled into the main event, Gradio's
        # loading overlay rendered as a stray floating box on this thin
        # bottom bar instead of staying inside the chat panel.
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
