use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};
use ratatui::{Frame, Terminal};

use crate::report::formatter::RenderOptions;
use crate::types::{CompareReport, ModelReport, ModelSourceKind};
use serde_json::Value;

const BAR_WIDTH: usize = 24;
const MAX_TENSORS: usize = 20;
const POLL_INTERVAL: Duration = Duration::from_millis(120);

const ACCENT_COLOR: Color = Color::Rgb(100, 180, 255);
const HEADER_COLOR: Color = Color::Rgb(130, 200, 255);
const MUTED_COLOR: Color = Color::Rgb(100, 100, 120);
const VALUE_COLOR: Color = Color::Rgb(220, 220, 240);
const BAR_EMPTY_COLOR: Color = Color::Rgb(50, 50, 60);
const WARN_COLOR: Color = Color::Rgb(255, 200, 80);
const CHANGED_COLOR: Color = Color::Rgb(255, 120, 100);
const SAME_COLOR: Color = Color::Rgb(100, 100, 110);
const TAB_ACTIVE_COLOR: Color = Color::Rgb(255, 210, 80);
const TAB_INACTIVE_COLOR: Color = Color::Rgb(140, 140, 160);
const BORDER_COLOR: Color = Color::Rgb(60, 65, 80);
const PANEL_BG: Color = Color::Rgb(18, 18, 28);
const FOOTER_COLOR: Color = Color::Rgb(80, 80, 100);

#[derive(Debug, Clone)]
struct TuiSection {
    title: String,
    lines: Vec<Line<'static>>,
}

#[derive(Debug, Clone)]
struct SectionApp {
    title: String,
    sections: Vec<TuiSection>,
    active_tab: usize,
    scroll: u16,
}

impl SectionApp {
    fn new(title: String, mut sections: Vec<TuiSection>) -> Self {
        if sections.is_empty() {
            sections.push(TuiSection {
                title: "Empty".to_string(),
                lines: vec![Line::from(Span::styled(
                    "No data available.",
                    Style::default().fg(MUTED_COLOR),
                ))],
            });
        }
        Self {
            title,
            sections,
            active_tab: 0,
            scroll: 0,
        }
    }

    fn active_section(&self) -> &TuiSection {
        &self.sections[self.active_tab]
    }

    fn next_tab(&mut self) {
        if self.sections.is_empty() {
            return;
        }
        self.active_tab = (self.active_tab + 1) % self.sections.len();
        self.scroll = 0;
    }

    fn prev_tab(&mut self) {
        if self.sections.is_empty() {
            return;
        }
        self.active_tab = if self.active_tab == 0 {
            self.sections.len().saturating_sub(1)
        } else {
            self.active_tab.saturating_sub(1)
        };
        self.scroll = 0;
    }

    fn scroll_down(&mut self, amount: u16) {
        self.scroll = self.scroll.saturating_add(amount);
    }

    fn scroll_up(&mut self, amount: u16) {
        self.scroll = self.scroll.saturating_sub(amount);
    }
}

struct TerminalGuard {
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
}

impl TerminalGuard {
    fn enter() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(err.into());
        }

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(t) => t,
            Err(err) => {
                let mut recover = io::stdout();
                let _ = execute!(recover, LeaveAlternateScreen);
                let _ = disable_raw_mode();
                return Err(err.into());
            }
        };
        terminal.clear()?;
        Ok(Self { terminal })
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

pub fn run_model_tui(report: &ModelReport, options: &RenderOptions) -> Result<()> {
    let sections = build_model_sections(report, options);
    run_tui_loop(SectionApp::new(
        format!("Model Report: {}", report.model),
        sections,
    ))
}

pub fn run_compare_tui(report: &CompareReport) -> Result<()> {
    let app = build_compare_app(report);
    run_compare_loop(app)
}

fn run_tui_loop(mut app: SectionApp) -> Result<()> {
    let mut guard = TerminalGuard::enter()?;

    loop {
        guard.terminal.draw(|frame| draw(frame, &app))?;
        if !event::poll(POLL_INTERVAL)? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press && key.kind != KeyEventKind::Repeat {
                continue;
            }

            let should_exit = match key.code {
                KeyCode::Esc | KeyCode::Char('q') => true,
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => {
                    app.next_tab();
                    false
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    app.prev_tab();
                    false
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    app.scroll_down(1);
                    false
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    app.scroll_up(1);
                    false
                }
                KeyCode::PageDown | KeyCode::Char(' ') => {
                    app.scroll_down(8);
                    false
                }
                KeyCode::PageUp => {
                    app.scroll_up(8);
                    false
                }
                KeyCode::Home => {
                    app.scroll = 0;
                    false
                }
                KeyCode::End => {
                    app.scroll_down(1000);
                    false
                }
                _ => false,
            };

            if should_exit {
                break;
            }
        }
    }

    Ok(())
}

fn draw(frame: &mut Frame<'_>, app: &SectionApp) {
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(frame.area());

    draw_title(frame, areas[0], app);
    draw_tabs(frame, areas[1], app);
    draw_content(frame, areas[2], app);
    draw_footer(frame, areas[3]);
}

fn draw_title(frame: &mut Frame<'_>, area: Rect, app: &SectionApp) {
    let title = Paragraph::new(app.title.clone())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER_COLOR))
                .title(Span::styled(
                    " DissectLM ",
                    Style::default()
                        .fg(ACCENT_COLOR)
                        .add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().bg(PANEL_BG)),
        )
        .alignment(Alignment::Center)
        .style(
            Style::default()
                .fg(HEADER_COLOR)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(title, area);
}

fn draw_tabs(frame: &mut Frame<'_>, area: Rect, app: &SectionApp) {
    let tab_labels: Vec<Line<'_>> = app
        .sections
        .iter()
        .enumerate()
        .map(|(i, s)| {
            if i == app.active_tab {
                Line::from(Span::styled(
                    s.title.clone(),
                    Style::default()
                        .fg(TAB_ACTIVE_COLOR)
                        .add_modifier(Modifier::BOLD),
                ))
            } else {
                Line::from(Span::styled(
                    s.title.clone(),
                    Style::default().fg(TAB_INACTIVE_COLOR),
                ))
            }
        })
        .collect();

    let tabs = Tabs::new(tab_labels)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER_COLOR))
                .title(Span::styled(
                    " Sections ",
                    Style::default().fg(MUTED_COLOR),
                ))
                .style(Style::default().bg(PANEL_BG)),
        )
        .divider(Span::styled(" │ ", Style::default().fg(BORDER_COLOR)))
        .select(app.active_tab)
        .highlight_style(
            Style::default()
                .fg(TAB_ACTIVE_COLOR)
                .add_modifier(Modifier::BOLD),
        );
    frame.render_widget(tabs, area);
}

fn draw_content(frame: &mut Frame<'_>, area: Rect, app: &SectionApp) {
    let section = app.active_section();

    let content = Paragraph::new(Text::from(section.lines.clone()))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER_COLOR))
                .title(Span::styled(
                    format!(" {} ", section.title),
                    Style::default()
                        .fg(ACCENT_COLOR)
                        .add_modifier(Modifier::BOLD),
                ))
                .style(Style::default().bg(PANEL_BG)),
        )
        .wrap(Wrap { trim: false })
        .scroll((app.scroll, 0));
    frame.render_widget(content, area);
}

fn draw_footer(frame: &mut Frame<'_>, area: Rect) {
    let controls = vec![
        Span::styled("q", Style::default().fg(ACCENT_COLOR).bold()),
        Span::styled(" quit ", Style::default().fg(FOOTER_COLOR)),
        Span::styled("│", Style::default().fg(BORDER_COLOR)),
        Span::styled(" ←/→", Style::default().fg(ACCENT_COLOR).bold()),
        Span::styled(" tabs ", Style::default().fg(FOOTER_COLOR)),
        Span::styled("│", Style::default().fg(BORDER_COLOR)),
        Span::styled(" j/k", Style::default().fg(ACCENT_COLOR).bold()),
        Span::styled(" scroll ", Style::default().fg(FOOTER_COLOR)),
        Span::styled("│", Style::default().fg(BORDER_COLOR)),
        Span::styled(" PgUp/PgDn", Style::default().fg(ACCENT_COLOR).bold()),
        Span::styled(" fast scroll", Style::default().fg(FOOTER_COLOR)),
    ];

    let footer = Paragraph::new(Line::from(controls))
        .block(
            Block::default()
                .borders(Borders::TOP)
                .border_style(Style::default().fg(BORDER_COLOR))
                .style(Style::default().bg(PANEL_BG)),
        )
        .alignment(Alignment::Center);
    frame.render_widget(footer, area);
}

fn build_model_sections(report: &ModelReport, options: &RenderOptions) -> Vec<TuiSection> {
    let mut sections = Vec::new();

    sections.push(TuiSection {
        title: "Summary".to_string(),
        lines: vec![
            kv_line("Model", &report.model),
            kv_line("Source kind", &source_kind_label(&report.source.kind)),
            kv_line("Source", &report.source.location),
            kv_line(
                "Total params (excl head)",
                &human_params(report.params.total_params),
            ),
            kv_line("Tensor files", &report.tensor_files_found.to_string()),
            kv_line(
                "Model size",
                &report
                    .model_size_bytes
                    .map(human_bytes)
                    .unwrap_or_else(|| "-".to_string()),
            ),
            kv_line(
                "Tensor dtypes",
                &if report.tensor_dtypes.is_empty() {
                    "-".to_string()
                } else {
                    report.tensor_dtypes.join(", ")
                },
            ),
            kv_line("Config keys", &report.config_key_count.to_string()),
            kv_line("Tensors indexed", &report.tensor_count.to_string()),
        ],
    });

    let distribution_rows = vec![
        (
            "FeedForward",
            report.params.categories.feedforward,
            report.params.pct(report.params.categories.feedforward),
        ),
        (
            "Attention",
            report.params.categories.attention,
            report.params.pct(report.params.categories.attention),
        ),
        (
            "Embedding",
            report.params.categories.embedding,
            report.params.pct(report.params.categories.embedding),
        ),
        (
            "Normalization",
            report.params.categories.normalization,
            report.params.pct(report.params.categories.normalization),
        ),
        (
            "OutputHead",
            report.params.categories.output_head,
            report.params.pct(report.params.categories.output_head),
        ),
        (
            "Other",
            report.params.categories.other,
            report.params.pct(report.params.categories.other),
        ),
    ];

    let mut distribution_lines = vec![
        Line::from(vec![
            Span::styled(
                format!("{:<14}", "Category"),
                Style::default().fg(MUTED_COLOR).bold(),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:>8}", "Share"),
                Style::default().fg(MUTED_COLOR).bold(),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:>14}", "Params"),
                Style::default().fg(MUTED_COLOR).bold(),
            ),
            Span::raw("  "),
            Span::styled("Distribution", Style::default().fg(MUTED_COLOR).bold()),
        ]),
        Line::from(Span::styled(
            "─".repeat(72),
            Style::default().fg(BORDER_COLOR),
        )),
    ];

    for (name, count, pct) in distribution_rows {
        let bar = pct_bar_styled(pct);
        let mut spans = vec![
            Span::styled(
                format!("{:<14}", name),
                Style::default().fg(VALUE_COLOR),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:>7.1}%", pct),
                Style::default().fg(pct_bar_color(pct)),
            ),
            Span::raw(" "),
            Span::styled(
                format!("{:>14}", human_params(count)),
                Style::default().fg(ACCENT_COLOR),
            ),
            Span::raw("  "),
        ];
        spans.extend(bar);
        distribution_lines.push(Line::from(spans));
    }

    sections.push(TuiSection {
        title: "Distribution".to_string(),
        lines: distribution_lines,
    });

    sections.push(TuiSection {
        title: "Architecture".to_string(),
        lines: {
            let mut lines = vec![
                kv_line("Layers", &opt_u64(report.architecture.num_layers)),
                kv_line("Hidden size", &opt_u64(report.architecture.hidden_size)),
                kv_line("Heads", &opt_u64(report.architecture.num_heads)),
                kv_line(
                    "KV heads",
                    &opt_u64(
                        report
                            .architecture
                            .num_key_value_heads
                            .or(report.attention.kv_heads),
                    ),
                ),
                kv_line(
                    "Attention type",
                    report
                        .architecture
                        .attention_type
                        .as_deref()
                        .or(report.attention.attention_type.as_deref())
                        .unwrap_or("-"),
                ),
            ];

            if let Some(config) = &report.config {
                for (key, value) in flatten_config_fields("cfg", config) {
                    lines.push(kv_line(&key, &value));
                }
            }

            lines
        },
    });

    if options.show_attention_breakdown {
        sections.push(TuiSection {
            title: "Attention".to_string(),
            lines: vec![
                kv_line(
                    "Q proj params",
                    &human_params(report.attention.q_proj_params),
                ),
                kv_line(
                    "K proj params",
                    &human_params(report.attention.k_proj_params),
                ),
                kv_line(
                    "V proj params",
                    &human_params(report.attention.v_proj_params),
                ),
                kv_line(
                    "O proj params",
                    &human_params(report.attention.o_proj_params),
                ),
            ],
        });
    }

    if options.show_graph {
        let lines = report
            .graph
            .as_deref()
            .unwrap_or("Graph unavailable")
            .lines()
            .map(|s| style_graph_line(s))
            .collect::<Vec<_>>();
        sections.push(TuiSection {
            title: "Graph".to_string(),
            lines,
        });
    }

    if options.show_params {
        let mut lines = vec![
            Line::from(vec![
                Span::styled(
                    format!("{:<4}", "#"),
                    Style::default().fg(MUTED_COLOR).bold(),
                ),
                Span::raw(" "),
                Span::styled(
                    format!("{:<62}", "Tensor"),
                    Style::default().fg(MUTED_COLOR).bold(),
                ),
                Span::raw(" "),
                Span::styled(
                    format!("{:>14}", "Params"),
                    Style::default().fg(MUTED_COLOR).bold(),
                ),
            ]),
            Line::from(Span::styled(
                "─".repeat(84),
                Style::default().fg(BORDER_COLOR),
            )),
        ];

        if let Some(tensors) = &report.tensors {
            let mut sorted = tensors.clone();
            sorted.sort_by(|a, b| b.param_count.cmp(&a.param_count));
            for (idx, tensor) in sorted.iter().take(MAX_TENSORS).enumerate() {
                lines.push(Line::from(vec![
                    Span::styled(
                        format!("{:<4}", idx + 1),
                        Style::default().fg(MUTED_COLOR),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<62}", truncate(&tensor.name, 62)),
                        Style::default().fg(VALUE_COLOR),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:>14}", human_params(tensor.param_count)),
                        Style::default().fg(ACCENT_COLOR),
                    ),
                ]));
            }
        } else {
            lines.push(Line::from(Span::styled(
                "Tensor details are unavailable.",
                Style::default().fg(MUTED_COLOR).italic(),
            )));
        }
        sections.push(TuiSection {
            title: "Top Tensors".to_string(),
            lines,
        });
    }

    if let Some(deep) = &report.deep {
        let deep_text = serde_json::to_string_pretty(deep).unwrap_or_else(|_| deep.to_string());
        let lines = deep_text
            .lines()
            .map(|s| {
                Line::from(Span::styled(
                    s.to_string(),
                    Style::default().fg(VALUE_COLOR),
                ))
            })
            .collect::<Vec<_>>();
        sections.push(TuiSection {
            title: "Deep".to_string(),
            lines,
        });
    }

    if !report.warnings.is_empty() {
        sections.push(TuiSection {
            title: "Warnings".to_string(),
            lines: report
                .warnings
                .iter()
                .map(|w| {
                    Line::from(Span::styled(
                        format!("! {w}"),
                        Style::default().fg(WARN_COLOR),
                    ))
                })
                .collect::<Vec<_>>(),
        });
    }

    sections
}

// ── Compare TUI: column-wise side-by-side layout ──

enum CompareTab {
    Dual { left: TuiSection, right: TuiSection },
    Single(TuiSection),
}

impl CompareTab {
    fn title(&self) -> &str {
        match self {
            CompareTab::Dual { left, .. } => &left.title,
            CompareTab::Single(s) => &s.title,
        }
    }
}

struct CompareApp {
    title: String,
    tabs: Vec<CompareTab>,
    active_tab: usize,
    scroll: u16,
}

impl CompareApp {
    fn next_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active_tab = (self.active_tab + 1) % self.tabs.len();
            self.scroll = 0;
        }
    }
    fn prev_tab(&mut self) {
        if !self.tabs.is_empty() {
            self.active_tab = if self.active_tab == 0 {
                self.tabs.len().saturating_sub(1)
            } else {
                self.active_tab.saturating_sub(1)
            };
            self.scroll = 0;
        }
    }
    fn scroll_down(&mut self, n: u16) { self.scroll = self.scroll.saturating_add(n); }
    fn scroll_up(&mut self, n: u16) { self.scroll = self.scroll.saturating_sub(n); }
}

fn run_compare_loop(mut app: CompareApp) -> Result<()> {
    let mut guard = TerminalGuard::enter()?;
    loop {
        guard.terminal.draw(|frame| draw_compare(frame, &app))?;
        if !event::poll(POLL_INTERVAL)? { continue; }
        if let Event::Key(key) = event::read()? {
            if key.kind != KeyEventKind::Press && key.kind != KeyEventKind::Repeat { continue; }
            let exit = match key.code {
                KeyCode::Esc | KeyCode::Char('q') => true,
                KeyCode::Right | KeyCode::Char('l') | KeyCode::Tab => { app.next_tab(); false }
                KeyCode::Left | KeyCode::Char('h') => { app.prev_tab(); false }
                KeyCode::Down | KeyCode::Char('j') => { app.scroll_down(1); false }
                KeyCode::Up | KeyCode::Char('k') => { app.scroll_up(1); false }
                KeyCode::PageDown | KeyCode::Char(' ') => { app.scroll_down(8); false }
                KeyCode::PageUp => { app.scroll_up(8); false }
                KeyCode::Home => { app.scroll = 0; false }
                KeyCode::End => { app.scroll_down(1000); false }
                _ => false,
            };
            if exit { break; }
        }
    }
    Ok(())
}

fn draw_compare(frame: &mut Frame<'_>, app: &CompareApp) {
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(1),
            Constraint::Length(2),
        ])
        .split(frame.area());

    // Title
    let title = Paragraph::new(app.title.clone())
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER_COLOR))
                .title(Span::styled(" DissectLM ", Style::default().fg(ACCENT_COLOR).add_modifier(Modifier::BOLD)))
                .style(Style::default().bg(PANEL_BG)),
        )
        .alignment(Alignment::Center)
        .style(Style::default().fg(HEADER_COLOR).add_modifier(Modifier::BOLD));
    frame.render_widget(title, areas[0]);

    // Tabs
    let tab_labels: Vec<Line<'_>> = app.tabs.iter().enumerate().map(|(i, t)| {
        let s = if i == app.active_tab {
            Style::default().fg(TAB_ACTIVE_COLOR).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(TAB_INACTIVE_COLOR)
        };
        Line::from(Span::styled(t.title().to_string(), s))
    }).collect();
    let tabs = Tabs::new(tab_labels)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(BORDER_COLOR))
                .title(Span::styled(" Sections ", Style::default().fg(MUTED_COLOR)))
                .style(Style::default().bg(PANEL_BG)),
        )
        .divider(Span::styled(" │ ", Style::default().fg(BORDER_COLOR)))
        .select(app.active_tab)
        .highlight_style(Style::default().fg(TAB_ACTIVE_COLOR).add_modifier(Modifier::BOLD));
    frame.render_widget(tabs, areas[1]);

    // Content — dual or single
    match &app.tabs[app.active_tab] {
        CompareTab::Dual { left, right } => {
            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                .split(areas[2]);

            let lp = Paragraph::new(Text::from(left.lines.clone()))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(BORDER_COLOR))
                        .title(Span::styled(format!(" {} ", left.title), Style::default().fg(ACCENT_COLOR).add_modifier(Modifier::BOLD)))
                        .style(Style::default().bg(PANEL_BG)),
                )
                .wrap(Wrap { trim: false })
                .scroll((app.scroll, 0));
            frame.render_widget(lp, cols[0]);

            let rp = Paragraph::new(Text::from(right.lines.clone()))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(BORDER_COLOR))
                        .title(Span::styled(format!(" {} ", right.title), Style::default().fg(ACCENT_COLOR).add_modifier(Modifier::BOLD)))
                        .style(Style::default().bg(PANEL_BG)),
                )
                .wrap(Wrap { trim: false })
                .scroll((app.scroll, 0));
            frame.render_widget(rp, cols[1]);
        }
        CompareTab::Single(section) => {
            let p = Paragraph::new(Text::from(section.lines.clone()))
                .block(
                    Block::default()
                        .borders(Borders::ALL)
                        .border_style(Style::default().fg(BORDER_COLOR))
                        .title(Span::styled(format!(" {} ", section.title), Style::default().fg(ACCENT_COLOR).add_modifier(Modifier::BOLD)))
                        .style(Style::default().bg(PANEL_BG)),
                )
                .wrap(Wrap { trim: false })
                .scroll((app.scroll, 0));
            frame.render_widget(p, areas[2]);
        }
    }

    draw_footer(frame, areas[3]);
}

fn build_compare_app(report: &CompareReport) -> CompareApp {
    let tabs = build_compare_tabs(report);
    CompareApp {
        title: "Model Comparison".to_string(),
        tabs,
        active_tab: 0,
        scroll: 0,
    }
}

fn build_compare_tabs(report: &CompareReport) -> Vec<CompareTab> {
    let mut tabs = Vec::new();
    let changed_count = report.diffs.iter().filter(|d| d.left != d.right).count();
    let total_count = report.diffs.len();

    // ── Summary (dual) ──
    let left_summary = vec![
        kv_line("Source", &source_kind_label(&report.left.source.kind)),
        kv_line("Params", &human_params(report.left.params.total_params)),
        kv_line("Model size", &report.left.model_size_bytes.map(human_bytes).unwrap_or_else(|| "-".into())),
        kv_line("Tensor files", &report.left.tensor_files_found.to_string()),
        kv_line("Tensors", &report.left.tensor_count.to_string()),
        kv_line("Dtypes", &if report.left.tensor_dtypes.is_empty() { "-".into() } else { report.left.tensor_dtypes.join(", ") }),
        kv_line("Config keys", &report.left.config_key_count.to_string()),
    ];
    let right_summary = vec![
        kv_line("Source", &source_kind_label(&report.right.source.kind)),
        kv_line("Params", &human_params(report.right.params.total_params)),
        kv_line("Model size", &report.right.model_size_bytes.map(human_bytes).unwrap_or_else(|| "-".into())),
        kv_line("Tensor files", &report.right.tensor_files_found.to_string()),
        kv_line("Tensors", &report.right.tensor_count.to_string()),
        kv_line("Dtypes", &if report.right.tensor_dtypes.is_empty() { "-".into() } else { report.right.tensor_dtypes.join(", ") }),
        kv_line("Config keys", &report.right.config_key_count.to_string()),
    ];
    tabs.push(CompareTab::Dual {
        left: TuiSection { title: report.left.model.clone(), lines: left_summary },
        right: TuiSection { title: report.right.model.clone(), lines: right_summary },
    });

    // ── Architecture (dual) ──
    let make_arch_lines = |r: &ModelReport| -> Vec<Line<'static>> {
        vec![
            kv_line("Model type", r.architecture.model_type.as_deref().unwrap_or("-")),
            kv_line("Layers", &opt_u64(r.architecture.num_layers)),
            kv_line("Hidden size", &opt_u64(r.architecture.hidden_size)),
            kv_line("Heads", &opt_u64(r.architecture.num_heads)),
            kv_line("KV heads", &opt_u64(r.architecture.num_key_value_heads.or(r.attention.kv_heads))),
            kv_line("Attention", r.architecture.attention_type.as_deref().or(r.attention.attention_type.as_deref()).unwrap_or("-")),
        ]
    };
    tabs.push(CompareTab::Dual {
        left: TuiSection { title: report.left.model.clone(), lines: make_arch_lines(&report.left) },
        right: TuiSection { title: report.right.model.clone(), lines: make_arch_lines(&report.right) },
    });

    // ── Distribution (dual with bars) ──
    let make_dist_lines = |r: &ModelReport| -> Vec<Line<'static>> {
        let categories: Vec<(&str, u64, f64)> = vec![
            ("FeedForward", r.params.categories.feedforward, r.params.pct(r.params.categories.feedforward)),
            ("Attention", r.params.categories.attention, r.params.pct(r.params.categories.attention)),
            ("Embedding", r.params.categories.embedding, r.params.pct(r.params.categories.embedding)),
            ("Normalization", r.params.categories.normalization, r.params.pct(r.params.categories.normalization)),
            ("OutputHead", r.params.categories.output_head, r.params.pct(r.params.categories.output_head)),
            ("Other", r.params.categories.other, r.params.pct(r.params.categories.other)),
        ];
        let mut lines = vec![
            Line::from(vec![
                Span::styled(format!("{:<14}", "Category"), Style::default().fg(MUTED_COLOR).bold()),
                Span::raw(" "),
                Span::styled(format!("{:>8}", "Share"), Style::default().fg(MUTED_COLOR).bold()),
                Span::raw(" "),
                Span::styled(format!("{:>12}", "Params"), Style::default().fg(MUTED_COLOR).bold()),
            ]),
            Line::from(Span::styled("─".repeat(40), Style::default().fg(BORDER_COLOR))),
        ];
        for (name, count, pct) in &categories {
            let bar = pct_bar_styled(*pct);
            let spans = vec![
                Span::styled(format!("{:<14}", name), Style::default().fg(VALUE_COLOR)),
                Span::raw(" "),
                Span::styled(format!("{:>7.1}%", pct), Style::default().fg(pct_bar_color(*pct))),
                Span::raw(" "),
                Span::styled(format!("{:>12}", human_params(*count)), Style::default().fg(ACCENT_COLOR)),
            ];
            lines.push(Line::from(spans));
            let mut bar_spans = vec![Span::raw("  ")];
            bar_spans.extend(bar);
            lines.push(Line::from(bar_spans));
        }
        lines.push(Line::from(Span::styled("─".repeat(40), Style::default().fg(BORDER_COLOR))));
        let total: u64 = categories.iter().map(|(_, c, _)| c).sum();
        lines.push(Line::from(vec![
            Span::styled(format!("{:<14}", "TOTAL"), Style::default().fg(VALUE_COLOR).add_modifier(Modifier::BOLD)),
            Span::raw(" "),
            Span::styled(format!("{:>8}", ""), Style::default().fg(MUTED_COLOR)),
            Span::raw(" "),
            Span::styled(format!("{:>12}", human_params(total)), Style::default().fg(ACCENT_COLOR).add_modifier(Modifier::BOLD)),
        ]));
        lines
    };
    tabs.push(CompareTab::Dual {
        left: TuiSection { title: report.left.model.clone(), lines: make_dist_lines(&report.left) },
        right: TuiSection { title: report.right.model.clone(), lines: make_dist_lines(&report.right) },
    });

    // ── Changes (single) ──
    let (l_p, r_p) = (report.left.params.total_params, report.right.params.total_params);
    let mut change_lines: Vec<Line<'static>> = Vec::new();

    change_lines.push(Line::from(vec![
        Span::styled(format!("{:<20}", "Status"), Style::default().fg(MUTED_COLOR).bold()),
        Span::raw(" "),
        Span::styled(
            format!("{changed_count}"),
            if changed_count > 0 { Style::default().fg(CHANGED_COLOR).add_modifier(Modifier::BOLD) }
            else { Style::default().fg(Color::Rgb(120, 220, 130)) },
        ),
        Span::styled(format!(" / {total_count} metrics differ"), Style::default().fg(MUTED_COLOR)),
    ]));

    if l_p != r_p {
        let (arrow, abs) = if r_p > l_p { ("▲", r_p - l_p) } else { ("▼", l_p - r_p) };
        let pct = if l_p > 0 { (abs as f64 / l_p as f64) * 100.0 } else { 100.0 };
        let dc = if r_p > l_p { Color::Rgb(120, 220, 130) } else { CHANGED_COLOR };
        change_lines.push(Line::from(vec![
            Span::styled(format!("{:<20}", "Param delta"), Style::default().fg(MUTED_COLOR).bold()),
            Span::raw(" "),
            Span::styled(human_params(l_p), Style::default().fg(VALUE_COLOR)),
            Span::styled(" → ", Style::default().fg(MUTED_COLOR)),
            Span::styled(human_params(r_p), Style::default().fg(VALUE_COLOR)),
            Span::raw("  "),
            Span::styled(format!("{arrow} {} ({:.1}%)", human_params(abs), pct), Style::default().fg(dc).add_modifier(Modifier::BOLD)),
        ]));
    } else {
        change_lines.push(kv_line("Param delta", &format!("{} (identical)", human_params(l_p))));
    }

    if let (Some(ls), Some(rs)) = (report.left.model_size_bytes, report.right.model_size_bytes) {
        if ls != rs {
            let (arrow, abs) = if rs > ls { ("▲", rs - ls) } else { ("▼", ls - rs) };
            let dc = if rs > ls { Color::Rgb(120, 220, 130) } else { CHANGED_COLOR };
            change_lines.push(Line::from(vec![
                Span::styled(format!("{:<20}", "Size delta"), Style::default().fg(MUTED_COLOR).bold()),
                Span::raw(" "),
                Span::styled(human_bytes(ls), Style::default().fg(VALUE_COLOR)),
                Span::styled(" → ", Style::default().fg(MUTED_COLOR)),
                Span::styled(human_bytes(rs), Style::default().fg(VALUE_COLOR)),
                Span::raw("  "),
                Span::styled(format!("{arrow} {}", human_bytes(abs)), Style::default().fg(dc).add_modifier(Modifier::BOLD)),
            ]));
        } else {
            change_lines.push(kv_line("Size delta", &format!("{} (identical)", human_bytes(ls))));
        }
    }

    let changed_diffs: Vec<_> = report.diffs.iter().filter(|d| d.left != d.right).collect();
    if !changed_diffs.is_empty() {
        change_lines.push(Line::from(vec![]));
        change_lines.push(Line::from(Span::styled(
            format!("  ⚠ Changed Metrics ({})", changed_diffs.len()),
            Style::default().fg(WARN_COLOR).add_modifier(Modifier::BOLD),
        )));
        change_lines.push(Line::from(Span::styled("╌".repeat(60), Style::default().fg(WARN_COLOR).add_modifier(Modifier::DIM))));
        for d in &changed_diffs {
            change_lines.push(Line::from(vec![
                Span::styled(" ≠ ", Style::default().fg(CHANGED_COLOR).add_modifier(Modifier::BOLD)),
                Span::styled(format!("{:<18}", truncate(&d.metric, 18)), Style::default().fg(Color::Rgb(255, 220, 100)).add_modifier(Modifier::BOLD)),
                Span::raw(" "),
                Span::styled(truncate(&d.left, 20), Style::default().fg(CHANGED_COLOR)),
                Span::styled(" → ", Style::default().fg(MUTED_COLOR)),
                Span::styled(truncate(&d.right, 20), Style::default().fg(Color::Rgb(120, 220, 130))),
            ]));
        }
    }

    let same_diffs: Vec<_> = report.diffs.iter().filter(|d| d.left == d.right).collect();
    if !same_diffs.is_empty() {
        change_lines.push(Line::from(vec![]));
        change_lines.push(Line::from(Span::styled(
            format!("  ✓ Identical Metrics ({})", same_diffs.len()),
            Style::default().fg(Color::Rgb(120, 220, 130)).add_modifier(Modifier::BOLD),
        )));
        change_lines.push(Line::from(Span::styled("╌".repeat(60), Style::default().fg(Color::Rgb(120, 220, 130)).add_modifier(Modifier::DIM))));
        for d in &same_diffs {
            change_lines.push(Line::from(vec![
                Span::styled(" · ", Style::default().fg(SAME_COLOR)),
                Span::styled(format!("{:<18}", truncate(&d.metric, 18)), Style::default().fg(VALUE_COLOR)),
                Span::raw(" "),
                Span::styled(truncate(&d.left, 24), Style::default().fg(SAME_COLOR)),
            ]));
        }
    }

    let mut all_warnings: Vec<String> = Vec::new();
    for w in &report.left.warnings { all_warnings.push(format!("[Left] {w}")); }
    for w in &report.right.warnings { all_warnings.push(format!("[Right] {w}")); }
    if !all_warnings.is_empty() {
        change_lines.push(Line::from(vec![]));
        change_lines.push(Line::from(Span::styled("  Warnings", Style::default().fg(WARN_COLOR).add_modifier(Modifier::BOLD))));
        for w in &all_warnings {
            change_lines.push(Line::from(Span::styled(format!("  ⚠ {w}"), Style::default().fg(WARN_COLOR))));
        }
    }

    tabs.push(CompareTab::Single(TuiSection {
        title: "Changes".to_string(),
        lines: change_lines,
    }));


    tabs
}


const GRAPH_BORDER_COLOR: Color = Color::Rgb(80, 130, 200);
const GRAPH_ARROW_COLOR: Color = Color::Rgb(100, 220, 255);
const GRAPH_TITLE_COLOR: Color = Color::Rgb(255, 220, 120);
const GRAPH_COMPONENT_COLOR: Color = Color::Rgb(180, 220, 255);
const GRAPH_DETAIL_COLOR: Color = Color::Rgb(120, 210, 160);
const GRAPH_REPEAT_COLOR: Color = Color::Rgb(80, 80, 120);
const GRAPH_BULLET_COLOR: Color = Color::Rgb(255, 180, 80);

/// Apply rich colorization to a single graph line.
///
/// Rules:
/// - Box-drawing characters (╭╮╯╰│├┤─) → blue border
/// - Arrows (▼) → cyan accent
/// - Repeat markers (┊) → muted dim
/// - Component bullets (○) → orange accent, rest of component text → light blue
/// - Parenthesized details (e.g., "(GQA, 32 heads)") → green
/// - Block titles (centered text inside boxes without bullets) → golden
fn style_graph_line(line: &str) -> Line<'static> {
    let trimmed = line.trim();

    // Pure arrow line
    if trimmed == "▼" {
        return Line::from(Span::styled(
            line.to_string(),
            Style::default()
                .fg(GRAPH_ARROW_COLOR)
                .add_modifier(Modifier::BOLD),
        ));
    }

    // Pure border line (only box-drawing + spaces)
    if is_pure_border(trimmed) {
        return Line::from(Span::styled(
            line.to_string(),
            Style::default().fg(GRAPH_BORDER_COLOR),
        ));
    }

    // Line with component bullets
    if trimmed.contains('○') {
        return style_component_line(line);
    }

    // Line with repeat markers and content
    if trimmed.starts_with('┊') {
        return style_repeated_content_line(line);
    }

    // Line with box borders and text content (title lines)
    if trimmed.starts_with('│') || trimmed.starts_with("┊") {
        return style_title_line(line);
    }

    // Fallback: plain styled
    Line::from(Span::styled(
        line.to_string(),
        Style::default().fg(VALUE_COLOR),
    ))
}

fn is_pure_border(s: &str) -> bool {
    s.chars().all(|c| {
        matches!(
            c,
            '╭' | '╮'
                | '╯'
                | '╰'
                | '│'
                | '├'
                | '┤'
                | '─'
                | '┊'
                | ' '
        )
    })
}

fn style_component_line(line: &str) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut chars = line.chars().peekable();
    let mut buffer = String::new();

    // Collect leading characters before the bullet
    while let Some(&c) = chars.peek() {
        if c == '○' {
            break;
        }
        buffer.push(c);
        chars.next();
    }

    // Style leading portion (repeat markers + borders)
    if !buffer.is_empty() {
        spans.push(Span::styled(
            buffer.clone(),
            Style::default().fg(GRAPH_BORDER_COLOR),
        ));
        buffer.clear();
    }

    // The bullet itself
    if chars.peek() == Some(&'○') {
        spans.push(Span::styled(
            "○".to_string(),
            Style::default()
                .fg(GRAPH_BULLET_COLOR)
                .add_modifier(Modifier::BOLD),
        ));
        chars.next();
    }

    // Remaining text after bullet
    let rest: String = chars.collect();

    // Split at parenthesis for detail highlighting
    if let Some(paren_start) = rest.find('(') {
        let before_paren = &rest[..paren_start];
        spans.push(Span::styled(
            before_paren.to_string(),
            Style::default().fg(GRAPH_COMPONENT_COLOR),
        ));

        if let Some(paren_end) = rest[paren_start..].find(')') {
            let paren_content = &rest[paren_start..paren_start + paren_end + 1];
            let after_paren = &rest[paren_start + paren_end + 1..];

            spans.push(Span::styled(
                paren_content.to_string(),
                Style::default().fg(GRAPH_DETAIL_COLOR),
            ));

            if !after_paren.is_empty() {
                // Trailing portion is likely box border chars
                spans.push(Span::styled(
                    after_paren.to_string(),
                    Style::default().fg(GRAPH_BORDER_COLOR),
                ));
            }
        } else {
            // No closing paren — just style the rest
            spans.push(Span::styled(
                rest[paren_start..].to_string(),
                Style::default().fg(GRAPH_DETAIL_COLOR),
            ));
        }
    } else {
        // No parens — split at trailing border chars
        let (text_part, border_part) = split_trailing_border(&rest);
        spans.push(Span::styled(
            text_part.to_string(),
            Style::default().fg(GRAPH_COMPONENT_COLOR),
        ));
        if !border_part.is_empty() {
            spans.push(Span::styled(
                border_part.to_string(),
                Style::default().fg(GRAPH_BORDER_COLOR),
            ));
        }
    }

    Line::from(spans)
}

fn style_repeated_content_line(line: &str) -> Line<'static> {
    // Lines like: "┊ │   title text   │  ┊"
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut chars = line.chars().peekable();
    let mut buffer = String::new();
    let mut in_text = false;
    let mut text_buf = String::new();

    while let Some(c) = chars.next() {
        if is_border_char(c) {
            if in_text && !text_buf.is_empty() {
                spans.push(Span::styled(
                    text_buf.clone(),
                    Style::default()
                        .fg(GRAPH_TITLE_COLOR)
                        .add_modifier(Modifier::BOLD),
                ));
                text_buf.clear();
                in_text = false;
            }
            buffer.push(c);
        } else {
            if !buffer.is_empty() {
                let color = if buffer.contains('┊') {
                    GRAPH_REPEAT_COLOR
                } else {
                    GRAPH_BORDER_COLOR
                };
                spans.push(Span::styled(buffer.clone(), Style::default().fg(color)));
                buffer.clear();
            }
            in_text = true;
            text_buf.push(c);
        }
    }

    // Flush remaining
    if !text_buf.is_empty() {
        spans.push(Span::styled(
            text_buf,
            Style::default()
                .fg(GRAPH_TITLE_COLOR)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if !buffer.is_empty() {
        let color = if buffer.contains('┊') {
            GRAPH_REPEAT_COLOR
        } else {
            GRAPH_BORDER_COLOR
        };
        spans.push(Span::styled(buffer, Style::default().fg(color)));
    }

    Line::from(spans)
}

fn style_title_line(line: &str) -> Line<'static> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut chars = line.chars().peekable();
    let mut buffer = String::new();
    let mut in_text = false;
    let mut text_buf = String::new();

    while let Some(c) = chars.next() {
        if is_border_char(c) {
            if in_text && !text_buf.is_empty() {
                spans.push(Span::styled(
                    text_buf.clone(),
                    Style::default()
                        .fg(GRAPH_TITLE_COLOR)
                        .add_modifier(Modifier::BOLD),
                ));
                text_buf.clear();
                in_text = false;
            }
            buffer.push(c);
        } else {
            if !buffer.is_empty() {
                spans.push(Span::styled(
                    buffer.clone(),
                    Style::default().fg(GRAPH_BORDER_COLOR),
                ));
                buffer.clear();
            }
            in_text = true;
            text_buf.push(c);
        }
    }

    if !text_buf.is_empty() {
        spans.push(Span::styled(
            text_buf,
            Style::default()
                .fg(GRAPH_TITLE_COLOR)
                .add_modifier(Modifier::BOLD),
        ));
    }
    if !buffer.is_empty() {
        spans.push(Span::styled(
            buffer,
            Style::default().fg(GRAPH_BORDER_COLOR),
        ));
    }

    Line::from(spans)
}

fn is_border_char(c: char) -> bool {
    matches!(
        c,
        '╭' | '╮'
            | '╯'
            | '╰'
            | '│'
            | '├'
            | '┤'
            | '─'
            | '┊'
    )
}

fn split_trailing_border(s: &str) -> (&str, &str) {
    // Find the last non-border character position
    let char_indices: Vec<(usize, char)> = s.char_indices().collect();
    let mut split_pos = s.len();

    for &(idx, c) in char_indices.iter().rev() {
        if is_border_char(c) || c == ' ' {
            split_pos = idx;
        } else {
            break;
        }
    }

    (&s[..split_pos], &s[split_pos..])
}

fn kv_line(key: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!("{:<20}", key),
            Style::default().fg(MUTED_COLOR).bold(),
        ),
        Span::raw(" "),
        Span::styled(value.to_string(), Style::default().fg(VALUE_COLOR)),
    ])
}

fn pct_bar_styled(pct: f64) -> Vec<Span<'static>> {
    let clamped = pct.clamp(0.0, 100.0);
    let filled_exact = (clamped / 100.0) * BAR_WIDTH as f64;
    let filled_full = filled_exact.floor() as usize;
    let remainder = filled_exact - filled_full as f64;

    let filled_full = filled_full.min(BAR_WIDTH);

    let partial_char = if remainder >= 0.75 {
        "▓"
    } else if remainder >= 0.5 {
        "▒"
    } else if remainder >= 0.25 {
        "░"
    } else {
        ""
    };

    let partial_count = if !partial_char.is_empty() && filled_full < BAR_WIDTH {
        1
    } else {
        0
    };

    let empty = BAR_WIDTH.saturating_sub(filled_full + partial_count);
    let bar_color = pct_bar_color(pct);

    let mut spans = Vec::new();
    if filled_full > 0 {
        spans.push(Span::styled(
            "█".repeat(filled_full),
            Style::default().fg(bar_color),
        ));
    }
    if !partial_char.is_empty() && partial_count > 0 {
        spans.push(Span::styled(
            partial_char.to_string(),
            Style::default().fg(bar_color),
        ));
    }
    if empty > 0 {
        spans.push(Span::styled(
            "·".repeat(empty),
            Style::default().fg(BAR_EMPTY_COLOR),
        ));
    }
    spans
}

fn pct_bar_color(pct: f64) -> Color {
    if pct >= 50.0 {
        Color::Rgb(100, 160, 255)
    } else if pct >= 25.0 {
        Color::Rgb(80, 200, 180)
    } else if pct >= 10.0 {
        Color::Rgb(120, 210, 130)
    } else if pct >= 5.0 {
        Color::Rgb(200, 210, 100)
    } else {
        Color::Rgb(100, 100, 110)
    }
}

fn human_params(value: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;
    const T: f64 = 1_000_000_000_000.0;

    let v = value as f64;
    if v >= T {
        format!("{:.2}T", v / T)
    } else if v >= B {
        format!("{:.2}B", v / B)
    } else if v >= M {
        format!("{:.2}M", v / M)
    } else if v >= K {
        format!("{:.2}K", v / K)
    } else {
        value.to_string()
    }
}

fn human_bytes(value: u64) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    const GB: f64 = 1024.0 * 1024.0 * 1024.0;
    const TB: f64 = 1024.0 * 1024.0 * 1024.0 * 1024.0;

    let v = value as f64;
    if v >= TB {
        format!("{:.2} TB", v / TB)
    } else if v >= GB {
        format!("{:.2} GB", v / GB)
    } else if v >= MB {
        format!("{:.2} MB", v / MB)
    } else if v >= KB {
        format!("{:.2} KB", v / KB)
    } else {
        format!("{value} B")
    }
}

fn source_kind_label(kind: &ModelSourceKind) -> String {
    match kind {
        ModelSourceKind::LocalPath => "local_path".to_string(),
        ModelSourceKind::HuggingFace => "hugging_face".to_string(),
    }
}

fn opt_u64(value: Option<u64>) -> String {
    value
        .map(|v| v.to_string())
        .unwrap_or_else(|| "-".to_string())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.chars().count() <= max_len {
        return s.to_string();
    }

    let mut out = s
        .chars()
        .take(max_len.saturating_sub(1))
        .collect::<String>();
    out.push('…');
    out
}

fn flatten_config_fields(prefix: &str, value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    flatten_config_fields_inner(prefix, value, &mut out);
    out
}

fn flatten_config_fields_inner(prefix: &str, value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            let mut keys = map.keys().collect::<Vec<_>>();
            keys.sort();
            for key in keys {
                if let Some(v) = map.get(key) {
                    flatten_config_fields_inner(&format!("{prefix}.{key}"), v, out);
                }
            }
        }
        Value::Array(arr) => {
            for (idx, v) in arr.iter().enumerate() {
                flatten_config_fields_inner(&format!("{prefix}[{idx}]"), v, out);
            }
        }
        _ => out.push((prefix.to_string(), format_config_leaf(value))),
    }
}

fn format_config_leaf(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Null => "null".to_string(),
        _ => serde_json::to_string(value).unwrap_or_else(|_| value.to_string()),
    }
}
