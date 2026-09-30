import os
import sys
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Image
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#718096"))

        # Suppress header and footer on cover page (page 1)
        if self._pageNumber > 1:
            # Header
            self.drawString(54, letter[1] - 36, "NAVRIS — Technical Research & Validation Report (SIH 2026)")
            self.drawRightString(letter[0] - 54, letter[1] - 36, "Phase 3.3B: Partially Supported")
            self.setStrokeColor(colors.HexColor("#CBD5E0"))
            self.setLineWidth(0.5)
            self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)

            # Footer
            self.line(54, 45, letter[0] - 54, 45)
            self.drawString(54, 32, "Confidential & Evaluator Review Copy | Smart India Hackathon 2026")
            self.drawRightString(letter[0] - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf(filename="docs/NAVRIS_Technical_Report.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    primary = colors.HexColor("#1A365D")    # Deep Navy
    secondary = colors.HexColor("#2B6CB0")  # Slate Blue
    dark_text = colors.HexColor("#2D3748")  # Charcoal
    border_color = colors.HexColor("#E2E8F0")

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=primary,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=secondary,
        spaceAfter=12
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=12,
        textColor=dark_text,
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=primary,
        spaceBefore=14,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=secondary,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12.2,
        textColor=dark_text,
        spaceAfter=5
    )

    callout_style = ParagraphStyle(
        'Callout_Text',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#2C5282")
    )

    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
        textColor=dark_text
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=dark_text
    )

    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.white
    )

    eq_style = ParagraphStyle(
        'EqStyle',
        parent=styles['Normal'],
        fontName='Courier-Bold',
        fontSize=7.8,
        leading=11,
        textColor=colors.HexColor("#1A202C")
    )

    story = []

    def callout(text, bg="#EBF8FF", border="#3182CE"):
        p = Paragraph(text, callout_style)
        t = Table([[p]], colWidths=[letter[0] - 108])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor(bg)),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor(border)),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        return t

    def equation_box(eq_text):
        p = Paragraph(eq_text, eq_style)
        t = Table([[p]], colWidths=[letter[0] - 108])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
            ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#E2E8F0")),
            ('LEFTPADDING', (0,0), (-1,-1), 12),
            ('RIGHTPADDING', (0,0), (-1,-1), 12),
            ('TOPPADDING', (0,0), (-1,-1), 5),
            ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ]))
        return t

    # -------------------------------------------------------------
    # COVER / HEADER
    # -------------------------------------------------------------
    story.append(Paragraph("NAVRIS", title_style))
    story.append(Paragraph("Intelligent Navigation & Inertial System: Technical Research & Validation Report", subtitle_style))
    story.append(Paragraph(
        "<b>Smart India Hackathon (SIH) 2026</b> &nbsp;|&nbsp; <b>Problem Statement ID:</b> 26168<br/>"
        "<b>Organization:</b> Indian Space Research Organisation (ISRO) / Department of Space<br/>"
        "<b>Theme:</b> Smart Vehicles &nbsp;|&nbsp; <b>Category:</b> Software<br/>"
        "<b>Lead Navigation Researcher:</b> Jay Patel (@the-jaypatel) &nbsp;|&nbsp; <b>Frontend Collaborator:</b> Rudra Patel (@ptlrudra0)<br/>"
        "<b>Primary Repository:</b> https://github.com/the-jaypatel/NAVRIS &nbsp;|&nbsp; <b>Frontend Repository:</b> https://github.com/ptlrudra0/TBA<br/>"
        "<b>Current Research Status:</b> Phase 3.3B — PARTIALLY SUPPORTED &nbsp;|&nbsp; <b>Deterministic Test Suite:</b> 176 Passed", meta_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary, spaceBefore=4, spaceAfter=10))

    # -------------------------------------------------------------
    # 1. Executive Summary
    # -------------------------------------------------------------
    story.append(Paragraph("1. Executive Summary", h1_style))
    story.append(Paragraph(
        "Modern intelligent transportation systems and autonomous vehicles depend critically on continuous, high-integrity positioning. "
        "While Global Navigation Satellite Systems (GNSS) provide absolute geographic fixes under nominal conditions, satellite signals are highly "
        "susceptible to multipath interference, urban canyon shadowing, tunnel dropouts, and atmospheric or intentional disruptions. "
        "NAVRIS (Intelligent Navigation & Inertial System) explores an AI/ML-assisted kinematic navigation architecture designed to maintain dead reckoning "
        "continuity during GNSS degradation and temporary outages using commodity smartphone micro-electro-mechanical (MEMS) inertial sensors.", body_style
    ))
    story.append(Paragraph(
        "The project enforces an uncompromising evidence-first scientific protocol. Unassisted double integration of low-cost smartphone accelerometers and "
        "gyroscopes suffers from rapid cubic error divergence exceeding hundreds of meters within twenty seconds. To counter this, NAVRIS builds a modular "
        "pipeline: (1) causal extrinsic calibration resolving gravity leveling and mounting yaw, (2) a mathematically audited 15-state continuous-discrete "
        "Error-State Extended Kalman Filter (ESKF), (3) causal kinematic constraints including Zero-Velocity Updates (ZUPT) and Non-Holonomic Constraints (NHC), "
        "and (4) an XGBoost forward-speed pseudo-measurement protected by an absolute reference firewall and two-stage innovation gating.", body_style
    ))
    story.append(Paragraph(
        "Across seven real-world vehicle driving sequences from the Oxford IO-VNBD dataset (spanning over 36,000 seconds of driving), the frozen causal ML "
        "pseudo-measurement reduced horizontal Root Mean Square Error (H-RMSE) relative to the frozen unassisted baseline across all seven routes. "
        "However, because absolute spatial errors remain substantial on extended driving sequences and performance varies by driving regime, the research "
        "status is scientifically classified as <b>PARTIALLY SUPPORTED</b>. Operational navigation accuracy, universal generalization, and vehicle platform independence "
        "have <b>not</b> been established.", body_style
    ))

    # -------------------------------------------------------------
    # 2. Problem Statement
    # -------------------------------------------------------------
    story.append(Paragraph("2. Problem Statement", h1_style))
    story.append(Paragraph(
        "In commercial aviation and marine transport, tactical-grade optical gyroscopes and precision accelerometers bound dead-reckoning drift. In automotive "
        "consumer contexts, however, navigators rely on commodity smartphone MEMS sensors costing cents per unit. These sensors suffer from severe non-linearities:<br/>"
        "• <b>Stochastic Sensor Drift:</b> Accelerometer biases (0.02 - 0.05 m/s²) and gyroscope thermal walk (0.001 rad/s) induce runaway cubic spatial divergence:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>Δp(t) ≈ 0.5 · b_a · t² + (1/6) · (b_g × g) · t³</b><br/>"
        "• <b>Arbitrary 3D Phone Mounting:</b> The device rests in arbitrary, uncalibrated orientations inside the cabin rather than aligned to vehicle chassis axes.<br/>"
        "• <b>Sparse, Asynchronous Fixes:</b> Mobile GNSS fixes arrive at irregular, low rates (~0.1 - 1 Hz) with high latency and sudden dropout boundaries.", body_style
    ))

    # -------------------------------------------------------------
    # 3. NAVRIS Concept
    # -------------------------------------------------------------
    story.append(Paragraph("3. NAVRIS Concept", h1_style))
    story.append(Paragraph(
        "Rather than treating navigation as an end-to-end black-box deep learning task—which notoriously hallucinates and fails unpredictably under out-of-distribution "
        "vehicle dynamics—NAVRIS maintains a strictly classical, physics-grounded filter at its core. Machine learning is restricted to a localized, well-bounded role: "
        "estimating forward longitudinal speed from causal trailing IMU windows during GNSS outages, and injecting it as a formal Kalman measurement update only when "
        "explicit reliability and innovation gates pass.", body_style
    ))

    # -------------------------------------------------------------
    # 4. System Architecture
    # -------------------------------------------------------------
    story.append(Paragraph("4. System Architecture", h1_style))
    story.append(Paragraph(
        "The complete end-to-end NAVRIS data processing and fusion flow is structured as follows:", body_style
    ))
    arch_box = (
        "<b>GNSS Telemetry + Smartphone IMU (10 Hz Synchronized Pipeline)</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Causal Extrinsic Calibration:</b> Gravity Leveling (Pitch/Roll) + Forward Acceleration Yaw Alignment<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Strapdown Inertial Propagation:</b> Zeroth-Order Quaternion Integration + Somigliana Normal Gravity<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>15-State Continuous-Discrete ESKF Core:</b> Van Loan Matrix Exponential Discretization (Φ, Q_d)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Causal Kinematic Constraints:</b> ZUPT (Stationary Velocity Nulling) + NHC (Lateral/Vertical Non-Slip)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Causal ML Forward-Speed Estimation:</b> W=10 trailing-window features, XGBoost Regressor<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Two-Stage Gating:</b> Reliability Classifier (P ≥ 0.70) + Innovation Gating (χ² ≤ 9.21)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>ML Forward-Speed Pseudo-Measurement Update:</b> Joseph Covariance Reset (R_ML = 0.50 m²/s²)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;↓<br/>"
        "<b>Navigation Output & Replay Bridge:</b> FastAPI Telemetry Server → TBA Cockpit Frontend"
    )
    story.append(callout(arch_box, bg="#F7FAFC", border="#CBD5E0"))

    # -------------------------------------------------------------
    # 5. IO-VNBD Dataset
    # -------------------------------------------------------------
    story.append(Paragraph("5. IO-VNBD Dataset", h1_style))
    story.append(Paragraph(
        "All empirical validation is performed using the Oxford Intelligent Orienting & Vehicle Navigation Benchmark Dataset (IO-VNBD). "
        "IO-VNBD comprises over 100 hours of synchronous vehicle telemetry recorded under diverse British driving regimes. "
        "Ground truth is provided by an RTK-grade OxTS RT3000 / VBOX inertial-GNSS reference system. The evaluated recordings include:<br/>"
        "• <b>S1 (5,018 s):</b> Suburban commute with arterial cruising, roundabouts, and signalized halts.<br/>"
        "• <b>S2 (9,191 s):</b> Prolonged continuous high-speed motorway cruise with minimal stationary events.<br/>"
        "• <b>S3A (2,171 s):</b> Dense urban corridor with repeated stop-and-go intervals.<br/>"
        "• <b>S4 (9,290 s):</b> Complex mixed suburban and arterial route with multi-point maneuvers.<br/>"
        "• <b>Y1 (7,176 s):</b> Highway cruise where the smartphone was mounted rotated ~106° relative to the vehicle frame.<br/>"
        "• <b>VTA1A (2,489 s):</b> Continuous highway cruise serving as an empirical negative control for ZUPT.<br/>"
        "• <b>VTA2 (1,013 s):</b> Suburban sequence featuring sharp 90-degree cornering transitions.", body_style
    ))

    # -------------------------------------------------------------
    # 6. Dataset Forensics
    # -------------------------------------------------------------
    story.append(Paragraph("6. Dataset Forensics: Resolving Critical Empirical Defects", h1_style))
    story.append(Paragraph(
        "In Phase 0, forensic auditing of 564 raw IO-VNBD telemetry files uncovered critical data corruption risks:<br/>"
        "1. <b>The 3.6× Velocity Scaling Bug:</b> Raw reference speed telemetry in `vbox_sync.csv` was logged in km/h while column metadata claimed m/s. "
        "Naive evaluation inflated velocity errors by 360%. NAVRIS resolved this via unit sanitization and verification against geodetic differentiation.<br/>"
        "2. <b>Temporal Lag & Desynchronization:</b> Smartphone IMU logs drifted up to 120 ms relative to reference timestamps. A causal 10 Hz cross-correlation "
        "pipeline was built to align clocks without future-time leakage.<br/>"
        "3. <b>Android vs Vehicle Frame Conventions:</b> Coordinate conventions between Android IMU standards (X-right, Y-up, Z-out) and vehicle navigation "
        "standards (X-forward, Y-right, Z-down) were formally mapped through explicit transformation matrices.", body_style
    ))

    # -------------------------------------------------------------
    # 7. Sensor Processing and Calibration
    # -------------------------------------------------------------
    story.append(Paragraph("7. Sensor Processing and Calibration", h1_style))
    story.append(Paragraph(
        "Because consumer smartphones rest in arbitrary orientations, body-frame dead reckoning requires online extrinsic alignment. "
        "NAVRIS implements a two-stage causal calibration protocol:<br/>"
        "• <b>Method A (Causal Gravity Leveling):</b> During standstill intervals identified by variance nulling, the mean specific force measures the gravity vector "
        "<b>g_b</b>, yielding pitch (θ) and roll (φ):<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;θ = arctan(-f_x / √(f_y² + f_z²)), &nbsp;&nbsp; φ = arctan(f_y / f_z)<br/>"
        "• <b>Method B (Causal Mounting Yaw Alignment):</b> During subsequent forward acceleration phases (where longitudinal acceleration dominates), horizontal acceleration "
        "is projected against heading changes to isolate mounting yaw (ψ). No future data is accessed.", body_style
    ))

    # -------------------------------------------------------------
    # 8. Inertial Dead Reckoning
    # -------------------------------------------------------------
    story.append(Paragraph("8. Inertial Dead Reckoning Formulation", h1_style))
    story.append(Paragraph(
        "Navigation states are mechanized in a local East-North-Up (ENU) geodetic tangent plane. True kinematic states evolve as:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>ṗ = v, &nbsp;&nbsp;&nbsp; v̇ = C_b_e · (f_m - b_a - w_a) + g_e, &nbsp;&nbsp;&nbsp; q̇ = 0.5 · q ⊗ [0, ω_m - b_g - w_g]^T</b><br/>"
        "where <b>g_e = [0, 0, -γ(φ)]^T</b> is evaluated using the WGS-84 Somigliana normal gravity model at vehicle latitude φ.", body_style
    ))

    # -------------------------------------------------------------
    # 9. ESKF Design
    # -------------------------------------------------------------
    story.append(Paragraph("9. 15-State Error-State Kalman Filter Design", h1_style))
    story.append(Paragraph(
        "The filter tracks nominal state <b>x̂</b> and error state <b>δx = [δp, δv, δθ, δb_a, δb_g]^T ∈ ℝ¹⁵</b>. "
        "Continuous-time error dynamics <b>d/dt(δx) = F_c δx + G_c w</b> are discretized via the Van Loan method, forming the block upper-triangular matrix exponential:", body_style
    ))
    story.append(equation_box(
        "A = [ -F_c,  G_c Q_c G_c^T ;  0,  F_c^T ] · Δt  ==&gt;  exp(A) = [ ..., Φ⁻¹ Q_d ; 0, Φ^T ]"
    ))
    story.append(Paragraph(
        "Joseph-form covariance updates ensure positive-semi-definiteness even under ill-conditioned measurement updates: "
        "<b>P_+ = (I - K H) P_- (I - K H)^T + K R K^T</b>.", body_style
    ))

    # -------------------------------------------------------------
    # 10. ZUPT
    # -------------------------------------------------------------
    story.append(Paragraph("10. Causal Zero-Velocity Updates (ZUPT)", h1_style))
    story.append(Paragraph(
        "During detected stationary periods, vehicle velocity in the navigation frame is zero: <b>z_zupt = v_e = 0</b> with measurement matrix <b>H_zupt = [0₃×₃, I₃×₃, 0₃×₉]</b>. "
        "The detector monitors a strictly trailing 1.0 s sliding window of accelerometer variance (σ_a² < 0.05 m²/s⁴) and gyroscope energy (||ω|| < 0.03 rad/s). "
        "On recording VTA2, ZUPT successfully arrested velocity drift during traffic stops, reducing horizontal RMSE by 99.63% (from 12.5 km down to 46.2 m).", body_style
    ))

    # -------------------------------------------------------------
    # 11. NHC
    # -------------------------------------------------------------
    story.append(Paragraph("11. Non-Holonomic Constraints (NHC)", h1_style))
    story.append(Paragraph(
        "Non-Holonomic Constraints model standard land vehicle kinematics: barring lateral side-slip and vertical vehicle liftoff, velocity in the vehicle body frame "
        "is constrained to the forward longitudinal axis: <b>v_y ≈ 0, v_z ≈ 0</b>. The measurement model is:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>z_nhc = [v_y, v_z]^T = [C_e_v]_2,3 · v_e + v_{nhc}</b><br/>"
        "with measurement Jacobian <b>H_nhc = [0₂×₃, [C_e_v]_2,3, -[C_e_v]_2,3 · [v_e ×], 0₂×₆]</b> and noise covariance <b>R_nhc = diag(0.20, 0.20) m²/s²</b>. "
        "NHC bounded cruising drift on highway route VTA1A by 34.52% without any stationary intervals.", body_style
    ))

    # -------------------------------------------------------------
    # 12. ML Forward-Speed Estimation
    # -------------------------------------------------------------
    story.append(Paragraph("12. Causal ML Forward-Speed Estimation", h1_style))
    story.append(Paragraph(
        "While NHC constrains lateral and vertical velocity, forward speed v_x remains unconstrained during GNSS outages. "
        "NAVRIS trains an XGBoost gradient-boosted regressor to predict forward speed <b>v̂_f</b> strictly from trailing causal IMU windows (W = 10 samples = 1.0 s). "
        "Feature engineering extracts 38 features: summary statistics (mean, variance, skew, kurtosis), jerk derivatives, spectral energy ratios, and tilt estimates. "
        "Supervision is drawn from VBOX reference speed during training, isolated behind an absolute firewall during test inference.", body_style
    ))

    # -------------------------------------------------------------
    # 13. Reliability Gating
    # -------------------------------------------------------------
    story.append(Paragraph("13. Reliability & Statistical Innovation Gating", h1_style))
    story.append(Paragraph(
        "To prevent anomalous ML predictions from corrupting the Kalman filter, a rigorous two-stage firewall is applied:<br/>"
        "1. <b>ML Reliability Classifier:</b> An auxiliary gradient-boosted classifier assesses feature distribution similarity and outputs a reliability probability P. "
        "Measurements with <b>P(reliable) < 0.70</b> are pruned.<br/>"
        "2. <b>Mahalanobis NIS Gating:</b> The filter computes the Normalized Innovation Squared against innovation covariance S: "
        "<b>NIS = ν^T S⁻¹ ν ≤ χ²_{2, 0.99} = 9.21</b>. Statistical outliers are rejected.", body_style
    ))

    # -------------------------------------------------------------
    # 14. ML Pseudo-Measurement
    # -------------------------------------------------------------
    story.append(Paragraph("14. ML Pseudo-Measurement Mechanism", h1_style))
    story.append(Paragraph(
        "Predictions passing both gates form a 3-DOF body velocity update: <b>z = [v̂_f, 0, 0]^T</b> with fixed measurement covariance "
        "<b>R_ML = diag(0.50, 0.10, 0.10) m²/s²</b>. Covariance is intentionally kept fixed and conservative to prevent filter overconfidence.", body_style
    ))

    # -------------------------------------------------------------
    # 15. Experimental Methodology
    # -------------------------------------------------------------
    story.append(Paragraph("15. Experimental Methodology & Rigor", h1_style))
    story.append(Paragraph(
        "All experiments strictly observe: (1) driver/route isolation across train, validation, and held-out test splits, (2) complete baseline code freezes before ML testing, "
        "and (3) automated deterministic verification. A 176-test suite covers every mathematical transformation, Jacobian derivation, and data boundary.", body_style
    ))

    # -------------------------------------------------------------
    # 16. Phase 3.3A
    # -------------------------------------------------------------
    story.append(Paragraph("16. Phase 3.3A: Controlled Pseudo-Measurement Audit", h1_style))
    story.append(Paragraph(
        "Phase 3.3A audited the pseudo-measurement mechanism on two initial routes (highway Y1 and suburban VTA2). "
        "The scientific audit verified analytical Jacobian exactness, zero future-leakage, and numerical stability across 8,000+ updates.", body_style
    ))

    # -------------------------------------------------------------
    # 17. Phase 3.3B
    # -------------------------------------------------------------
    story.append(Paragraph("17. Phase 3.3B: Broader Generalization Benchmark", h1_style))
    story.append(Paragraph(
        "Phase 3.3B extended the evaluation across all seven IO-VNBD driving routes without tuning hyperparameters, thresholds, or covariances. "
        "The frozen model and filter were evaluated strictly as pre-declared.", body_style
    ))

    # -------------------------------------------------------------
    # 18. Results
    # -------------------------------------------------------------
    story.append(Paragraph("18. Results & Quantitative Benchmark Evaluation", h1_style))
    story.append(Paragraph(
        "The table below presents the verified horizontal position RMSE (H-RMSE) across all seven evaluated IO-VNBD driving recordings, "
        "comparing the frozen baseline (Arm A: ESKF + ZUPT + NHC) against the ML-augmented filter (Arm C: Baseline + ML Pseudo-Measurement):", body_style
    ))

    results_data = [
        [Paragraph("Recording", table_header),
         Paragraph("Domain", table_header),
         Paragraph("Duration", table_header),
         Paragraph("Baseline H-RMSE (m)", table_header),
         Paragraph("Arm C H-RMSE (m)", table_header),
         Paragraph("Δ vs Base (%)", table_header),
         Paragraph("Empirical Classification", table_header)],
        [Paragraph("S1", table_cell_bold), Paragraph("Train", table_cell), Paragraph("5,018 s", table_cell),
         Paragraph("11,580,767.64", table_cell), Paragraph("2,230.98", table_cell), Paragraph("-99.98%", table_cell), Paragraph("Large relative reduction", table_cell)],
        [Paragraph("S2", table_cell_bold), Paragraph("Train", table_cell), Paragraph("9,191 s", table_cell),
         Paragraph("42,389,910.67", table_cell), Paragraph("17,646,092.96", table_cell), Paragraph("-58.37%", table_cell), Paragraph("High rejection (54.7% NIS rej)", table_cell)],
        [Paragraph("S3A", table_cell_bold), Paragraph("Validation", table_cell), Paragraph("2,171 s", table_cell),
         Paragraph("620,666.46", table_cell), Paragraph("9,310.67", table_cell), Paragraph("-98.50%", table_cell), Paragraph("Substantial reduction", table_cell)],
        [Paragraph("S4", table_cell_bold), Paragraph("Train", table_cell), Paragraph("9,290 s", table_cell),
         Paragraph("37,351,981.92", table_cell), Paragraph("3,060.08", table_cell), Paragraph("-99.99%", table_cell), Paragraph("Large relative reduction", table_cell)],
        [Paragraph("Y1", table_cell_bold), Paragraph("Held-out Test", table_cell), Paragraph("7,176 s", table_cell),
         Paragraph("29,884,134.11", table_cell), Paragraph("3,209,429.27", table_cell), Paragraph("-89.26%", table_cell), Paragraph("Reduced, large residual error", table_cell)],
        [Paragraph("VTA1A", table_cell_bold), Paragraph("Validation", table_cell), Paragraph("2,489 s", table_cell),
         Paragraph("1,480,088.21", table_cell), Paragraph("21,683.94", table_cell), Paragraph("-98.54%", table_cell), Paragraph("Substantial reduction", table_cell)],
        [Paragraph("VTA2", table_cell_bold), Paragraph("Held-out Test", table_cell), Paragraph("1,013 s", table_cell),
         Paragraph("31,080.62", table_cell), Paragraph("4,793.30", table_cell), Paragraph("-84.58%", table_cell), Paragraph("Bounded spatial error", table_cell)],
    ]

    col_widths = [52, 65, 48, 95, 92, 65, 87]
    results_table = Table(results_data, colWidths=col_widths)
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), secondary),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#CBD5E0")),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#F7FAFC")]),
    ]))
    story.append(results_table)
    story.append(Spacer(1, 8))

    # Add diagnostic visual evidence if plot exists
    plot_path = "docs/plots/gate2_3b/s3a_trajectory_comparison.png"
    if os.path.exists(plot_path):
        story.append(Paragraph("<b>Empirical Trajectory Evidence (S3A Route):</b>", h2_style))
        img = Image(plot_path, width=4.8*inch, height=3.6*inch)
        story.append(img)
        story.append(Paragraph("<i>Figure 1: Ground truth reference (OxTS RT3000 VBOX) vs unassisted baseline vs constrained solution on S3A.</i>", meta_style))

    # -------------------------------------------------------------
    # 19. Scientific Interpretation
    # -------------------------------------------------------------
    story.append(Paragraph("19. Scientific Interpretation", h1_style))
    story.append(Paragraph(
        "<b>Important Caution on Relative Reductions:</b> Relative reductions (e.g., -58% to -99%) are computed against unassisted frozen baselines "
        "that diverge by tens of thousands of kilometers due to cumulative double integration of sensor biases. "
        "<b>They should not be interpreted as operational navigation accuracy.</b> "
        "Absolute position errors of 2.2 km (S1), 3.1 km (S4), 4.8 km (VTA2), and over 3,200 km (Y1) and 17,646 km (S2) clearly demonstrate that unassisted "
        "dead reckoning with consumer smartphone sensors cannot replace GNSS over long durations without absolute external fixes.", body_style
    ))
    story.append(Paragraph(
        "<b>Regime Dependence on Recording S2:</b> S2 exhibited poor innovation consistency: 54.67% of proposed ML updates were rejected by the filter's innovation gate, "
        "with an applied innovation RMSE of 743.69 m/s. This empirical outcome highlights significant sensitivity to high-speed motorway driving regimes where subtle "
        "sensor calibration drift compounds over extended duration.", body_style
    ))

    # -------------------------------------------------------------
    # 20. Limitations
    # -------------------------------------------------------------
    story.append(Paragraph("20. Technical Limitations & Confounders", h1_style))
    story.append(Paragraph(
        "• <b>No Universal Accuracy Claim:</b> Bounded errors (<5 m) are achieved only on short routes with frequent stationary stops (VTA2).<br/>"
        "• <b>No Vehicle Independence:</b> Validated on British passenger cars; heavy trucks, motorcycles, and rail dynamics remain unproven.<br/>"
        "• <b>No Real-Time Edge Hardware:</b> Evaluated in Python offline research/replay mode; native Android on-device execution is not yet implemented.", body_style
    ))

    # -------------------------------------------------------------
    # 21. Current Implementation & Demo
    # -------------------------------------------------------------
    story.append(Paragraph("21. Current Implementation & Demo Integration", h1_style))
    story.append(Paragraph(
        "The repository contains an operational, runnable software stack:<br/>"
        "• <b>Core Navigation Library:</b> High-performance Python implementation in `src/navris/` covering coordinates, calibration, ESKF, ZUPT, NHC, and ML modules.<br/>"
        "• <b>FastAPI Replay Backend:</b> A REST API (`src/navris/api/main.py`) serving verified IO-VNBD benchmark replay frames and audit telemetry over `/api/v1/nav/*`.<br/>"
        "• <b>Cockpit Frontend (TBA):</b> A standalone React/TypeScript cockpit (developed and maintained by Rudra Patel at `ptlrudra0/TBA`) connecting to the FastAPI "
        "backend, supporting both client-side simulation demo and NAVRIS research/replay visualization.", body_style
    ))

    # -------------------------------------------------------------
    # 22. Future Work
    # -------------------------------------------------------------
    story.append(Paragraph("22. Future Work (Phase 4 Roadmap)", h1_style))
    story.append(Paragraph(
        "1. <b>Adaptive Innovation-Based Covariance:</b> Dynamically scaling R_ML based on trailing innovation variance rather than fixed covariance.<br/>"
        "2. <b>Temporal Convolutional Networks:</b> Evaluating causal 1D-CNN/TCN architectures to extract higher-order kinematic inertial features.<br/>"
        "3. <b>Embedded Android C++ Engine:</b> Porting the 15-state ESKF and quantized ONNX ML runtime to native Android NDK for live on-device testing.", body_style
    ))

    # -------------------------------------------------------------
    # 23. Conclusion
    # -------------------------------------------------------------
    story.append(Paragraph("23. Scientific Conclusion", h1_style))
    conclusion_text = (
        "<b>Current Research Status: Phase 3.3B — PARTIALLY SUPPORTED</b><br/><br/>"
        "The frozen ML forward-speed pseudo-measurement reduced horizontal RMSE relative to the frozen baseline on all seven evaluated recordings, "
        "including the two held-out test-domain recordings. However, the magnitude and quality of the effect varied substantially by recording. "
        "S2 exhibited poor innovation consistency, and several long-duration recordings retained very large absolute spatial errors. "
        "The experiment therefore provides evidence that the causal pseudo-measurement can alter and sometimes reduce observed filter divergence, "
        "but it does not establish uniform generalization, operational navigation accuracy, vehicle/platform independence, or long-duration navigation stability."
    )
    story.append(callout(conclusion_text, bg="#FFFDF0", border="#D69E2E"))

    # -------------------------------------------------------------
    # 24. Repository and Frontend References
    # -------------------------------------------------------------
    story.append(Paragraph("24. Repository and Frontend References", h1_style))
    story.append(Paragraph(
        "• <b>Primary Research & Backend Repository:</b> <a href='https://github.com/the-jaypatel/NAVRIS'>https://github.com/the-jaypatel/NAVRIS</a> (Lead: Jay Patel)<br/>"
        "• <b>Cockpit Frontend Repository:</b> <a href='https://github.com/ptlrudra0/TBA'>https://github.com/ptlrudra0/TBA</a> (Collaborator: Rudra Patel)<br/>"
        "• <b>Benchmark Dataset:</b> Oxford Intelligent Orienting & Vehicle Navigation Benchmark Dataset (Oxford Robotics Institute)<br/>"
        "• <b>Reference Framework:</b> Sola, J., <i>Quaternion kinematics for the error-state Kalman filter</i>, arXiv:1711.02508.<br/>"
        "• <b>Smart India Hackathon (SIH) 2026:</b> Problem Statement 26168, Indian Space Research Organisation (ISRO) / Department of Space.", body_style
    ))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Report built successfully at: {filename}")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "docs/NAVRIS_Technical_Report.pdf"
    build_pdf(out_path)
