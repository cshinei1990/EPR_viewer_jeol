import os
import re
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

try:
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    from scipy.special import wofz
    from scipy.interpolate import UnivariateSpline
    from scipy.interpolate import InterpolatedUnivariateSpline
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

SAVE_JSON_PATH = "saved_spectra.json"


def parse_header_value(line):
    if '=' not in line:
        return None
    return line.split('=', 1)[1].strip()


def parse_jeol_file(path):
    text = None
    for enc in ("utf-8", "cp932", "shift_jis"):
        try:
            with open(path, "r", encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise UnicodeDecodeError("cannot decode file with tried encodings")

    lines = text.splitlines()

    header = {}
    in_data_section = False
    data_lines = []

    for line in lines:
        if line.startswith("===== Data Head"):
            continue

        if line.startswith("====== DATA"):
            in_data_section = True
            continue

        if not in_data_section:
            if line.startswith("file name"):
                header["file_name"] = parse_header_value(line)
            elif line.startswith("data length"):
                v = parse_header_value(line)
                header["data_length"] = int(v)
            elif line.startswith("x-range min"):
                v = parse_header_value(line)
                header["x_range_min"] = float(v)
            elif line.startswith("x-range     ="):
                v = parse_header_value(line)
                header["x_range"] = float(v)
            elif line.startswith("modulation freq."):
                header["modulation_freq"] = parse_header_value(line)
            elif line.startswith("mod. width(fine)"):
                header["mod_width_fine_raw"] = parse_header_value(line)
            elif line.startswith("mod. width(coarse)"):
                header["mod_width_coarse_raw"] = parse_header_value(line)
            elif line.startswith("amplitude(fine)"):
                header["amplitude_fine_raw"] = parse_header_value(line)
            elif line.startswith("amplitude(coarse)"):
                header["amplitude_coarse_raw"] = parse_header_value(line)
            elif line.startswith("micro power unit"):
                header["micro_power_unit_raw"] = parse_header_value(line)
            elif line.startswith("micro power"):
                header["micro_power_raw"] = parse_header_value(line)
            elif line.startswith("micro frequency"):
                header["micro_freq_raw"] = parse_header_value(line)
        else:
            data_lines.append(line)

    y_vals = []
    data_started = False
    for line in data_lines:
        if "Real part data" in line and "Index=0" in line:
            data_started = True
            continue
        if not data_started:
            continue

        pieces = line.strip().split()
        for p in pieces:
            try:
                y_vals.append(float(p))
            except ValueError:
                pass

    y = np.array(y_vals, dtype=float)

    data_length = header.get("data_length", len(y))
    x_min = header.get("x_range_min")
    x_range = header.get("x_range")

    if x_min is None or x_range is None:
        raise ValueError("x-range min または x-range がヘッダーから取得できませんでした。")

    if data_length <= 1:
        raise ValueError("data length が不正です。")

    dx = x_range / (data_length - 1)
    x = x_min + dx * np.arange(data_length)

    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    return header, x, y


def decode_md_am(raw_fine, raw_coarse, prefix):
    if raw_fine is None or raw_coarse is None:
        raise ValueError("fine / coarse がヘッダーに含まれていません。")

    m = re.match(rf"{prefix}([+-]?\d+\.?\d*)", raw_fine)
    if not m:
        raise ValueError(f"fine の書式が不明です: {raw_fine}")
    fine_val = float(m.group(1))

    m2 = re.match(rf"{prefix}([+-]?\d+\.?\d*)", raw_coarse)
    if not m2:
        raise ValueError(f"coarse の書式が不明です: {raw_coarse}")
    exp_val = float(m2.group(1))

    return fine_val * (10.0 ** exp_val)


def decode_micro_power(raw_power, raw_unit):
    if raw_power is None or raw_unit is None:
        raise ValueError("micro power または unit がヘッダーに含まれていません。")

    m = re.match(r"uP([+-]?\d+\.?\d*)", raw_power)
    if not m:
        raise ValueError(f"micro power の書式が不明です: {raw_power}")
    val = float(m.group(1))

    unit = raw_unit.upper()
    if "MW" in unit:
        factor = 1.0
    elif "UW" in unit:
        factor = 1.0 / 1000.0
    else:
        factor = 1.0
        print(f"Warning: unknown micro power unit: {raw_unit}")

    return val * factor  # [mW]


def decode_micro_freq(raw_freq):
    if raw_freq is None:
        raise ValueError("micro frequency がヘッダーに含まれていません。")
    m = re.match(r"uF([+-]?\d+\.?\d*)", raw_freq)
    if not m:
        raise ValueError(f"micro frequency の書式が不明です: {raw_freq}")
    return float(m.group(1))


def cumulative_trapezoid(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    if n < 2:
        return np.zeros_like(y)
    out = np.zeros_like(y, dtype=float)
    for i in range(1, n):
        dx = x[i] - x[i - 1]
        out[i] = out[i - 1] + 0.5 * (y[i] + y[i - 1]) * dx
    return out


def gaussian_absorption(x, amp, x0, fwhm):
    x = np.asarray(x)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amp * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))


def lorentzian_absorption(x, amp, x0, fwhm):
    x = np.asarray(x)
    gamma = fwhm / 2.0
    return amp * (gamma ** 2) / ((x - x0) ** 2 + gamma ** 2)


def voigt_absorption_true(x, amp, x0, fwhm, eta=None):
    x = np.asarray(x)
    if eta is None:
        eta = 0.0  # デフォルト 0.0 = 純 Lorentzian 寄り
    eta = max(0.0, min(1.0, eta))

    fwhm_g = eta * fwhm
    fwhm_l = (1.0 - eta) * fwhm

    sigma = fwhm_g / (2.0 * np.sqrt(2.0 * np.log(2.0))) if fwhm_g > 0 else 1e-9
    gamma = fwhm_l / 2.0 if fwhm_l > 0 else 1e-9

    x_shift = x - x0
    z = (x_shift + 1j * gamma) / (sigma * np.sqrt(2.0))
    V = np.real(wofz(z)) / (sigma * np.sqrt(2.0 * np.pi))
    return amp * V


def gaussian_derivative(x, amp, x0, fwhm):
    x = np.asarray(x)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    g = np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
    dgdx = g * (-(x - x0) / (sigma ** 2))
    return amp * dgdx


def lorentzian_derivative(x, amp, x0, fwhm):
    x = np.asarray(x)
    gamma = fwhm / 2.0
    denom = (x - x0) ** 2 + gamma ** 2
    dldx = amp * (-2.0 * (x - x0) * gamma ** 2) / (denom ** 2)
    return dldx


def voigt_derivative_true(x, amp, x0, fwhm, eta=None):
    if eta is None:
        eta = 0.0
    eps = fwhm / 100.0 if fwhm > 0 else 1e-3
    base_plus = voigt_absorption_true(x + eps, amp, x0, fwhm, eta)
    base_minus = voigt_absorption_true(x - eps, amp, x0, fwhm, eta)
    return (base_plus - base_minus) / (2.0 * eps)


def multi_peak_model(x, *params_and_types):
    """
    params_and_types: 1ピークあたり 5 つ
      [amp, x0, width, eta, t_idx]
    """
    x = np.asarray(x)
    y_total = np.zeros_like(x, dtype=float)

    for i in range(0, len(params_and_types), 5):
        if i + 4 >= len(params_and_types):
            break
        amp = params_and_types[i]
        x0 = params_and_types[i + 1]
        w = params_and_types[i + 2]
        eta = params_and_types[i + 3]
        t_idx = int(round(params_and_types[i + 4]))
        if w <= 0:
            continue

        if t_idx == 0:
            y_total += gaussian_absorption(x, amp, x0, w)
        elif t_idx == 1:
            y_total += lorentzian_absorption(x, amp, x0, w)
        elif t_idx == 2:
            y_total += voigt_absorption_true(x, amp, x0, w, eta)
        elif t_idx == 3:
            y_total += gaussian_derivative(x, amp, x0, w)
        elif t_idx == 4:
            y_total += lorentzian_derivative(x, amp, x0, w)
        elif t_idx == 5:
            y_total += voigt_derivative_true(x, amp, x0, w, eta)
        else:
            y_total += gaussian_absorption(x, amp, x0, w)
    return y_total


class EPRViewerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("JEOL EPR データビューア（Gauss・mW正規化・BL＆Voigtコンボ・マルチフィット版）")
        self.geometry("1220x900")

        self.current_header = None
        self.current_x = None
        self.current_y = None

        self.saved_spectra = []

        self.pointer_source_x = None
        self.pointer_source_y = None

        self.baseline_select_mode = False
        self.drag_start_x = None
        self.baseline_ranges = []
        self.baseline_spans = []

        self.peak_rows = []
        self.area_rows = []
        self.fit_target_index = None
        self.fit_window = None

        # 赤線ピークマーカー
        self.peak_centers = []
        self.peak_marker_artists = [] 
        # 計測カーソル関連
        self.cursor_mode = None  # None / "A" / "B"
        self.cursorA = None
        self.cursorB = None
        self.cursorA_artists = []
        self.cursorB_artists = []

        self.create_widgets()
        self.connect_matplotlib_events()
        self.load_saved_spectra_from_json()

    # ---------------- GUI ----------------

    def create_widgets(self):
        main_pane = tk.PanedWindow(self, orient=tk.VERTICAL)
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        upper_frame = tk.Frame(main_pane)
        bottom_frame = tk.Frame(main_pane)
        main_pane.add(upper_frame, minsize=400)
        main_pane.add(bottom_frame, minsize=260)
        self.main_pane = main_pane
        # 上側
        top_frame = tk.Frame(upper_frame)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        open_btn = tk.Button(
            top_frame,
            text="ファイル選択（補正して保存）",
            command=self.open_files_and_correct_save,
        )
        open_btn.pack(side=tk.LEFT)

        tk.Label(top_frame, text="file name:").pack(side=tk.LEFT, padx=(10, 2))
        self.file_label_var = tk.StringVar(value="-")
        tk.Label(top_frame, textvariable=self.file_label_var, width=40, anchor="w").pack(side=tk.LEFT)

        freq_frame = tk.Frame(upper_frame)
        freq_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(freq_frame, text="測定 micro frequency [MHz]:").pack(side=tk.LEFT)
        self.meas_freq_var = tk.StringVar(value="-")
        tk.Label(freq_frame, textvariable=self.meas_freq_var, width=12, anchor="w").pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(freq_frame, text="基準 micro frequency [MHz]:").pack(side=tk.LEFT)
        self.ref_freq_entry = tk.Entry(freq_frame, width=12)
        self.ref_freq_entry.pack(side=tk.LEFT)

        plot_btn = tk.Button(freq_frame, text="補正して表示（このファイル）", command=self.plot_current_spectrum)
        plot_btn.pack(side=tk.LEFT, padx=10)

        save_btn = tk.Button(freq_frame, text="補正して保存（このファイル）", command=self.save_corrected_spectrum)
        save_btn.pack(side=tk.LEFT)
        # === 上段：操作UI用フレーム ===
        control_frame = tk.Frame(upper_frame)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # --- ポインター表示 ---
        pointer_frame = tk.Frame(control_frame)
        pointer_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 2))
        self.pointer_var = tk.StringVar(value="X = -, Y = -")
        tk.Label(pointer_frame, text="ポインター:").pack(side=tk.LEFT)
        tk.Label(pointer_frame, textvariable=self.pointer_var, width=40, anchor="w").pack(side=tk.LEFT)

        # --- カーソル表示 ---
        cursor_frame = tk.Frame(control_frame)
        cursor_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        self.cursor_info_var = tk.StringVar(value="カーソルA/B: 未設定")
        tk.Label(cursor_frame, textvariable=self.cursor_info_var, anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        cursor_btn_frame = tk.Frame(control_frame)
        cursor_btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))
        tk.Label(cursor_btn_frame, text="計測カーソル:").pack(side=tk.LEFT)
        tk.Button(cursor_btn_frame, text="カーソルAセット", command=lambda: self.set_cursor_mode("A")).pack(side=tk.LEFT, padx=2)
        tk.Button(cursor_btn_frame, text="カーソルBセット", command=lambda: self.set_cursor_mode("B")).pack(side=tk.LEFT, padx=2)
        tk.Button(cursor_btn_frame, text="クリア", command=self.clear_cursors).pack(side=tk.LEFT, padx=2)

        # --- ベースライン操作 ---
        bl_frame = tk.Frame(control_frame)
        bl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Button(bl_frame, text="ベースライン範囲選択(ドラッグ)", command=self.toggle_baseline_selection).pack(side=tk.LEFT)
        tk.Button(bl_frame, text="範囲クリア", command=self.clear_baseline_ranges).pack(side=tk.LEFT, padx=5)
        tk.Button(bl_frame, text="プレビュー", command=self.preview_baseline_selected_spectrum).pack(side=tk.LEFT, padx=5)
        tk.Button(bl_frame, text="補正", command=self.baseline_correct_selected_spectrum).pack(side=tk.LEFT, padx=5)

        bl2_frame = tk.Frame(control_frame)
        bl2_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(bl2_frame, text="平滑ポイント:").pack(side=tk.LEFT)
        self.smooth_entry = tk.Entry(bl2_frame, width=5)
        self.smooth_entry.insert(0, "1")
        self.smooth_entry.pack(side=tk.LEFT)

        tk.Label(bl2_frame, text="基準レベル:").pack(side=tk.LEFT, padx=(10, 2))
        self.dest_entry = tk.Entry(bl2_frame, width=7)
        self.dest_entry.insert(0, "0.0")
        self.dest_entry.pack(side=tk.LEFT)

        tk.Label(bl2_frame, text="形状:").pack(side=tk.LEFT, padx=(10, 2))
        self.baseline_method_var = tk.StringVar(value="spline")
        tk.Radiobutton(bl2_frame, text="spline", variable=self.baseline_method_var, value="spline").pack(side=tk.LEFT)
        tk.Radiobutton(bl2_frame, text="spline_linear", variable=self.baseline_method_var, value="spline_linear").pack(side=tk.LEFT, padx=(0, 10))

        # --- ベースライン範囲一覧 ---
        bl_range_frame = tk.Frame(control_frame)
        bl_range_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(0, 5))

        tk.Label(bl_range_frame, text="範囲一覧:").pack(side=tk.LEFT)
        self.bl_range_listbox = tk.Listbox(bl_range_frame, height=3, width=40)
        self.bl_range_listbox.pack(side=tk.LEFT, padx=(5, 5))
        tk.Button(bl_range_frame, text="削除", command=self.remove_selected_baseline_range).pack(side=tk.LEFT)
        tk.Button(bl_range_frame, text="全消去", command=self.clear_baseline_ranges).pack(side=tk.LEFT, padx=5)

        # === 下段：グラフ描画用フレーム ===
        canvas_frame = tk.Frame(upper_frame)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        fig = plt.Figure(figsize=(7, 5))
        self.ax = fig.add_subplot(111)
        self.ax.grid(True)

        self.canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, canvas_frame)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 下側
        btn_frame = tk.Frame(bottom_frame)
        btn_frame.pack(fill=tk.X, pady=3)
        show_btn = tk.Button(btn_frame, text="選択スペクトルを表示", command=self.show_selected_spectra)
        show_btn.pack(side=tk.LEFT)

        delete_btn = tk.Button(btn_frame, text="選択スペクトルを削除", command=self.delete_selected_spectra)
        delete_btn.pack(side=tk.LEFT, padx=5)

        clear_btn = tk.Button(btn_frame, text="保存リストをすべてクリア", command=self.clear_saved_spectra)
        clear_btn.pack(side=tk.LEFT, padx=5)
        int_fit_frame = tk.Frame(bottom_frame)
        int_fit_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(3, 0))

        int_btn = tk.Button(int_fit_frame, text="選択スペクトルを積分", command=self.integrate_selected_spectrum)
        int_btn.pack(side=tk.LEFT)

        diff_btn = tk.Button(int_fit_frame, text="選択スペクトルを微分", command=self.differentiate_selected_spectrum)
        diff_btn.pack(side=tk.LEFT, padx=5)

        fit_btn = tk.Button(int_fit_frame, text="マルチピークフィット", command=self.open_multi_peak_fit_window)
        fit_btn.pack(side=tk.LEFT, padx=10)
        
        

        tk.Label(bottom_frame, text="保存スペクトル一覧（補正／BL補正／積分／微分／フィット後データ）").pack(anchor="w")

        list_frame = tk.Frame(bottom_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)

        self.spectrum_listbox = tk.Listbox(
            list_frame, selectmode=tk.EXTENDED, height=8, width=80
        )
        self.spectrum_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.spectrum_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.spectrum_listbox.config(yscrollcommand=scrollbar.set)

        

    # --------- Matplotlib events ---------

    def connect_matplotlib_events(self):
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def on_mouse_move(self, event):
        if event.inaxes != self.ax:
            return
        if self.pointer_source_x is None or self.pointer_source_y is None:
            return
        if event.xdata is None:
            return

        x_arr = self.pointer_source_x
        y_arr = self.pointer_source_y
        idx = int(np.argmin(np.abs(x_arr - event.xdata)))
        x_val = x_arr[idx]
        y_val = y_arr[idx]
        self.pointer_var.set(f"X = {x_val:.4f} G, Y = {y_val:.4g}")

    def on_mouse_press(self, event):
        # --- カーソルモードを最優先 ---
        if self.cursor_mode in ("A", "B") and event.inaxes == self.ax and event.xdata is not None:
            self._set_cursor_point_from_click(event)
            return
        # 1) ベースライン選択モード（ドラッグ開始）
        if self.baseline_select_mode:
            if event.inaxes != self.ax or event.xdata is None:
                return
            if event.button != 1:
                return
            self.drag_start_x = event.xdata
            return

        # 2) カーソルモード中なら、A/B の位置をセット
        if (
            self.cursor_mode in ("A", "B")
            and event.inaxes == self.ax
            and event.xdata is not None
            and event.button == 1
        ):
            self._set_cursor_point_from_click(event)
            return

        # 3) フィットウィンドウ開いている & カーソルモードでない場合 → クリックでピーク追加
        if (
            event.inaxes == self.ax
            and event.xdata is not None
            and event.button == 1
            and self.fit_window is not None
            and self.fit_window.winfo_exists()
            and self.fit_target_index is not None
            and self.cursor_mode is None
        ):
            # 対象スペクトルの x, y 上で最近傍点を取得
            spec = self.saved_spectra[self.fit_target_index]
            x = spec["x"]
            y = spec["y"]
            idx = int(np.argmin(np.abs(x - event.xdata)))
            center = float(x[idx])
            amp = float(y[idx])
            width_default = (x.max() - x.min()) / 50.0
            if width_default <= 0:
                width_default = 1.0

            # dVoigt で 1 ピーク行追加
            self._append_peak_row(center, amp, width_default, "dVoigt", eta=0.0)
            return

    def on_mouse_release(self, event):
        if not self.baseline_select_mode:
            return
        if self.drag_start_x is None:
            return
        if event.inaxes != self.ax or event.xdata is None:
            self.drag_start_x = None
            return
        if event.button != 1:
            self.drag_start_x = None
            return

        x1, x2 = sorted([self.drag_start_x, event.xdata])
        self.drag_start_x = None

        self.baseline_ranges.append((x1, x2))
        span = self.ax.axvspan(x1, x2, alpha=0.2)
        self.baseline_spans.append(span)

        # Listbox を更新
        if hasattr(self, "bl_range_listbox"):
            self._refresh_baseline_range_listbox()

        self.canvas.draw()

    # -------- JSON 永続化 --------

    def load_saved_spectra_from_json(self):
        if not os.path.exists(SAVE_JSON_PATH):
            return
        try:
            with open(SAVE_JSON_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return

        spectra_list = []
        ref_freq_value = None

        if isinstance(raw, dict):
            spectra_list = raw.get("spectra", [])
            ref_freq_value = raw.get("ref_freq", None)
        elif isinstance(raw, list):
            spectra_list = raw
        else:
            return

        for item in spectra_list:
            try:
                label = item.get("label", "unknown")
                x_list = item.get("x", [])
                y_list = item.get("y", [])
                x = np.array(x_list, dtype=float)
                y = np.array(y_list, dtype=float)
                spec = {"label": label, "x": x, "y": y}
                fit_meta = item.get("fit_meta")
                if fit_meta is not None:
                    spec["fit_meta"] = fit_meta
                self.saved_spectra.append(spec)
                self.spectrum_listbox.insert(tk.END, label)
            except Exception:
                continue

        if ref_freq_value is not None:
            try:
                val = float(ref_freq_value)
                self.ref_freq_entry.delete(0, tk.END)
                self.ref_freq_entry.insert(0, f"{val:.3f}")
            except Exception:
                pass

    def dump_saved_spectra_to_json(self):
        spectra_data = []
        for spec in self.saved_spectra:
            item = {
                "label": spec["label"],
                "x": spec["x"].tolist(),
                "y": spec["y"].tolist(),
            }
            if "fit_meta" in spec:
                item["fit_meta"] = spec["fit_meta"]
            spectra_data.append(item)

        try:
            ref_val = float(self.ref_freq_entry.get())
        except Exception:
            ref_val = None

        data_to_save = {
            "ref_freq": ref_val,
            "spectra": spectra_data,
        }

        try:
            with open(SAVE_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    def _append_or_replace_saved_spectrum(self, spec):
        """
        同じ label のスペクトルがあれば置き換え、
        なければ末尾に追加するヘルパー
        """
        label = spec.get("label")
        if label is None:
            return

        for i, s in enumerate(self.saved_spectra):
            if s.get("label") == label:
                # 既存を置き換え
                self.saved_spectra[i] = spec
                self.spectrum_listbox.delete(i)
                self.spectrum_listbox.insert(i, label)
                break
        else:
            # 新規追加
            self.saved_spectra.append(spec)
            self.spectrum_listbox.insert(tk.END, label)
    # -------- ファイル入力(単一/複数統合) --------

    def open_files_and_correct_save(self):
        paths = filedialog.askopenfilenames(
            title="JEOL EPR テキストファイルを選択（1つ以上）",
            filetypes=[("Text files", "*.txt;*.dat;*.asc;*.*"), ("All files", "*.*")]
        )
        if not paths:
            return

        parsed = []
        for path in paths:
            try:
                header, x, y = parse_jeol_file(path)
                parsed.append((header, x, y, path))
            except Exception as e:
                messagebox.showwarning("警告", f"{os.path.basename(path)} の読み込みに失敗しました:\n{e}")
        if not parsed:
            messagebox.showwarning("警告", "有効なファイルがありませんでした。")
            return

        try:
            ref_freq = float(self.ref_freq_entry.get())
        except ValueError:
            ref_freq = None

        if ref_freq is None:
            header0 = parsed[0][0]
            try:
                meas_freq = decode_micro_freq(header0.get("micro_freq_raw"))
            except Exception:
                meas_freq = None
            if meas_freq is None:
                messagebox.showerror("エラー", "基準 micro frequency を入力するか、ヘッダから取得できませんでした。")
                return
            ref_freq = meas_freq
            self.ref_freq_entry.delete(0, tk.END)
            self.ref_freq_entry.insert(0, f"{ref_freq:.3f}")

        added_count = 0
        last_spec = None

        for header, x_mT, y, path in parsed:
            try:
                x_corr_G, y_norm = self.compute_corrected_from_raw(header, x_mT, y, ref_freq)
            except Exception as e:
                messagebox.showwarning("警告", f"{os.path.basename(path)} の補正に失敗しました:\n{e}")
                continue

            file_name = header.get("file_name", os.path.basename(path))
            label = f"{file_name}  (ref {ref_freq:.3f} MHz)"
            self.saved_spectra.append({"label": label, "x": x_corr_G, "y": y_norm})
            self.spectrum_listbox.insert(tk.END, label)
            added_count += 1
            last_spec = (x_corr_G, y_norm, label)

            self.current_header = header
            self.current_x = x_mT
            self.current_y = y
            self.file_label_var.set(file_name)
            try:
                micro_freq_MHz = decode_micro_freq(header.get("micro_freq_raw"))
                self.meas_freq_var.set(f"{micro_freq_MHz:.3f}")
            except Exception:
                self.meas_freq_var.set("-")

        if added_count > 0 and last_spec is not None:
            self.dump_saved_spectra_to_json()
            x_corr_G, y_norm, label = last_spec
            self.clear_baseline_ranges(dont_message=True)
            self.ax.clear()
            self.ax.plot(x_corr_G, y_norm, linewidth=1.0)
            self.ax.set_xlabel(f"Magnetic field (G, corrected to {ref_freq:.3f} MHz)")
            self.ax.set_ylabel("Intensity (normalized arb. units)")
            self.ax.set_title(label)
            self.ax.grid(True)
            self.canvas.draw()
            self.pointer_source_x = x_corr_G
            self.pointer_source_y = y_norm
            messagebox.showinfo("読み込み＆保存", f"{added_count} 個のファイルを補正して保存しました。")

    def compute_corrected_from_raw(self, header, x_mT, y, ref_freq):
        md_mT = decode_md_am(header.get("mod_width_fine_raw"),
                             header.get("mod_width_coarse_raw"), "md")
        md_G = md_mT * 10.0
        amp = decode_md_am(header.get("amplitude_fine_raw"),
                           header.get("amplitude_coarse_raw"), "am")
        micro_power_mW = decode_micro_power(header.get("micro_power_raw"),
                                            header.get("micro_power_unit_raw"))

        denom = md_G * amp * (micro_power_mW ** 0.5)
        if denom == 0:
            raise ValueError("規格化の分母が 0 になります。")

        y_norm = y / denom

        meas_freq = decode_micro_freq(header.get("micro_freq_raw"))
        if meas_freq <= 0 or ref_freq <= 0:
            raise ValueError("micro frequency は正の値である必要があります。")

        scale = ref_freq / meas_freq
        x_corr_G = x_mT * 10.0 * scale
        return x_corr_G, y_norm

    def compute_corrected_spectrum_current(self):
        if self.current_header is None or self.current_x is None or self.current_y is None:
            raise ValueError("先にファイルを読み込んでください。")
        try:
            ref_freq = float(self.ref_freq_entry.get())
        except ValueError:
            raise ValueError("基準 micro frequency を正しく入力してください。")

        x_corr_G, y_norm = self.compute_corrected_from_raw(
            self.current_header, self.current_x, self.current_y, ref_freq
        )
        return x_corr_G, y_norm, ref_freq

    def plot_current_spectrum(self):
        try:
            x_corr_G, y_norm, ref_freq = self.compute_corrected_spectrum_current()
        except Exception as e:
            messagebox.showerror("エラー", f"補正計算に失敗しました:\n{e}")
            return

        self.clear_baseline_ranges(dont_message=True)

        self.ax.clear()
        self.ax.plot(x_corr_G, y_norm, linewidth=1.0)
        self.ax.set_xlabel(f"Magnetic field (G, corrected to {ref_freq:.3f} MHz)")
        self.ax.set_ylabel("Intensity (normalized arb. units)")
        self.ax.set_title(self.file_label_var.get() + "（再計算）")
        self.ax.grid(True)
        self.canvas.draw()

        self.pointer_source_x = x_corr_G
        self.pointer_source_y = y_norm
        self._draw_peak_markers()

    def save_corrected_spectrum(self):
        if self.current_header is None:
            messagebox.showwarning("警告", "先にファイルを読み込んでください。")
            return
        try:
            x_corr_G, y_norm, ref_freq = self.compute_corrected_spectrum_current()
        except Exception as e:
            messagebox.showerror("エラー", f"補正計算に失敗しました:\n{e}")
            return

        label = f"{self.file_label_var.get()}  (ref {ref_freq:.3f} MHz)"
        self.saved_spectra.append({"label": label, "x": x_corr_G, "y": y_norm})
        self.spectrum_listbox.insert(tk.END, label)
        self.dump_saved_spectra_to_json()
        messagebox.showinfo("保存完了", "補正後スペクトルを保存リストと JSON に追加しました。")

    # -------- 保存スペクトル表示・管理 --------

    def show_selected_spectra(self):
        selection = self.spectrum_listbox.curselection()
        if not selection:
            messagebox.showwarning("警告", "表示したいスペクトルをリストから選択してください。")
            return

        self.clear_baseline_ranges(dont_message=True)

        self.ax.clear()
        last_spec = None
        for idx in selection:
            spec = self.saved_spectra[idx]
            self.ax.plot(spec["x"], spec["y"], linewidth=1.0, label=spec["label"])
            last_spec = spec

        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity (normalized arb. units)")
        self.ax.grid(True)
        self.ax.legend(fontsize=8)
        self.ax.set_title("保存スペクトル")
        self.canvas.draw()

        if last_spec is not None:
            self.pointer_source_x = last_spec["x"]
            self.pointer_source_y = last_spec["y"]

        self._draw_peak_markers()

    def delete_selected_spectra(self):
        selection = list(self.spectrum_listbox.curselection())
        if not selection:
            messagebox.showwarning("警告", "削除したいスペクトルを選択してください。")
            return
        for idx in reversed(selection):
            del self.saved_spectra[idx]
            self.spectrum_listbox.delete(idx)
        self.dump_saved_spectra_to_json()
        messagebox.showinfo("削除", "選択したスペクトルを削除しました。")

    def clear_saved_spectra(self):
        self.saved_spectra.clear()
        self.spectrum_listbox.delete(0, tk.END)
        self.dump_saved_spectra_to_json()
        messagebox.showinfo("クリア", "保存スペクトル一覧と JSON をクリアしました。")

    # -------- ベースライン --------

    def toggle_baseline_selection(self):
        self.baseline_select_mode = not self.baseline_select_mode
        if self.baseline_select_mode:
            messagebox.showinfo(
                "ベースライン範囲選択",
                "プロット上で左クリック＋ドラッグして矩形範囲を指定してください。\n"
                "複数個指定できます。やめるときはもう一度このボタンか「ベースライン範囲クリア」を押してください。"
            )

    def clear_baseline_ranges(self, dont_message=False):
        self.baseline_ranges = []
        for span in self.baseline_spans:
            try:
                span.remove()
            except Exception:
                pass
        self.baseline_spans = []
        self.drag_start_x = None

        # Listbox もクリア
        if hasattr(self, "bl_range_listbox"):
            self.bl_range_listbox.delete(0, tk.END)

        self.canvas.draw()
        if not dont_message:
            messagebox.showinfo("クリア", "ベースライン範囲を全てクリアしました。")
    
    def _refresh_baseline_range_listbox(self):
        """self.baseline_ranges の内容を Listbox に反映"""
        if not hasattr(self, "bl_range_listbox"):
            return

        self.bl_range_listbox.delete(0, tk.END)
        for i, (x1, x2) in enumerate(self.baseline_ranges):
            self.bl_range_listbox.insert(
                tk.END,
                f"{i+1}: {x1:.3f} – {x2:.3f} G"
            )

    def remove_selected_baseline_range(self):
        """Listbox で選択されているベースライン範囲だけを削除"""
        if not self.baseline_ranges or not hasattr(self, "bl_range_listbox"):
            return

        sel = self.bl_range_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if 0 <= idx < len(self.baseline_ranges):
            self.baseline_ranges.pop(idx)

        # 既存の span を全部消してから、残っている範囲で描き直し
        for span in self.baseline_spans:
            try:
                span.remove()
            except Exception:
                pass
        self.baseline_spans = []
        for x1, x2 in self.baseline_ranges:
            span = self.ax.axvspan(x1, x2, alpha=0.2)
            self.baseline_spans.append(span)

        self._refresh_baseline_range_listbox()
        self.canvas.draw()

    def _compute_baseline_from_ranges(self, x, y):
        if not self.baseline_ranges:
            raise ValueError("ベースライン範囲が設定されていません。")

        mask_total = np.zeros_like(x, dtype=bool)
        for x1, x2 in self.baseline_ranges:
            mask_total |= (x >= x1) & (x <= x2)

        if not np.any(mask_total):
            raise ValueError("指定したベースライン範囲内にデータ点がありません。")

        x_base = x[mask_total]
        y_base = y[mask_total]

        sort_idx = np.argsort(x_base)
        x_base_sorted = x_base[sort_idx]
        y_base_sorted = y_base[sort_idx]

        if len(x_base_sorted) == 1:
            baseline_line = np.full_like(x, y_base_sorted[0])
        else:
            # ベースライン形状の選択（spline / spline_liner）
            method = getattr(self, "baseline_method_var", None)
            if isinstance(method, tk.StringVar):
                method = method.get()
            else:
                method = "spline"

            if method == "spline_linear":
                # 3次多項式フィット。点が少ない場合は次数を下げる
                spline = InterpolatedUnivariateSpline(x_base_sorted, y_base_sorted, k=1)
                baseline_line = spline(x)
            else:
                # spline: SciPy があれば UnivariateSpline、なければ線形補間
                if SCIPY_AVAILABLE:
                    try:
                        s_val = float(self.smooth_entry.get())
                    except:
                        s_val = 0.0

                    try:
                        # x の単調性確認
                        if np.any(np.diff(x_base_sorted) <= 0):
                            raise ValueError("x must be strictly increasing for spline")

                        if len(x_base_sorted) < 4:
                            raise ValueError("Not enough points for spline")

                        spline = UnivariateSpline(x_base_sorted, y_base_sorted, s=s_val)
                        baseline_line = spline(x)
                    except:
                        baseline_line = np.interp(x, x_base_sorted, y_base_sorted)
                else:
                    baseline_line = np.interp(x, x_base_sorted, y_base_sorted)

        return baseline_line, mask_total

    def preview_baseline_selected_spectrum(self):
        selection = self.spectrum_listbox.curselection()
        if len(selection) != 1:
            messagebox.showwarning("警告", "ベースラインをプレビューするスペクトルを 1 つだけ選択してください。")
            return
        if not self.baseline_ranges:
            messagebox.showwarning("警告", "先にドラッグでベースライン範囲を 1 つ以上指定してください。")
            return

        idx = selection[0]
        spec = self.saved_spectra[idx]
        x = spec["x"]
        y = spec["y"]

        try:
            baseline_line, _ = self._compute_baseline_from_ranges(x, y)
        except Exception as e:
            messagebox.showerror("エラー", f"ベースライン計算に失敗しました:\n{e}")
            return

        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.0, alpha=0.6, label="original")
        self.ax.plot(x, baseline_line, linestyle="--", linewidth=1.0, label="baseline (preview)")
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True)
        self.ax.set_title(spec["label"] + "  [BL preview]")
        self.ax.legend(fontsize=8)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y

        self._draw_peak_markers()

    def baseline_correct_selected_spectrum(self):
        selection = self.spectrum_listbox.curselection()
        if len(selection) != 1:
            messagebox.showwarning("警告", "ベースライン補正するスペクトルを 1 つだけ選択してください。")
            return
        if not self.baseline_ranges:
            messagebox.showwarning("警告", "先にドラッグでベースライン範囲を 1 つ以上指定してください。")
            return

        idx = selection[0]
        spec = self.saved_spectra[idx]
        x = spec["x"]
        y = spec["y"]

        try:
            baseline_line, mask_total = self._compute_baseline_from_ranges(x, y)
        except Exception as e:
            messagebox.showerror("エラー", f"ベースライン計算に失敗しました:\n{e}")
            return

        y_corr = y - baseline_line

        try:
            win = int(self.smooth_entry.get())
        except ValueError:
            win = 1
        if win < 1:
            win = 1
        if win > 1:
            if win % 2 == 0:
                win += 1
            kernel = np.ones(win) / win
            y_corr = np.convolve(y_corr, kernel, mode="same")

        try:
            dest_level = float(self.dest_entry.get())
        except ValueError:
            dest_level = 0.0

        y_corr_base_mean = np.mean(y_corr[mask_total])
        shift = dest_level - y_corr_base_mean
        y_corr = y_corr + shift

        spec["y"] = y_corr
        self.saved_spectra[idx] = spec
        self.dump_saved_spectra_to_json()

        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.0, alpha=0.3, label="original")
        self.ax.plot(x, baseline_line, linestyle="--", linewidth=1.0, label="baseline")
        self.ax.plot(x, y_corr, linewidth=1.0, label="corrected")
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity (baseline corrected)")
        self.ax.grid(True)
        self.ax.set_title(spec["label"] + "  [BL applied]")
        self.ax.legend(fontsize=8)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y_corr

        self._draw_peak_markers()

    # -------- 積分 --------

    def integrate_selected_spectrum(self):
        selection = self.spectrum_listbox.curselection()
        if len(selection) != 1:
            messagebox.showwarning("警告", "積分するスペクトルを 1 つだけ選択してください。")
            return
        idx = selection[0]
        spec = self.saved_spectra[idx]
        x = spec["x"]
        y = spec["y"]

        y_int = cumulative_trapezoid(x, y)

        label = spec["label"]
        ints = re.findall(r"\[Int(\d+)\]", label)
        if ints:
            max_n = max(int(n) for n in ints)
            new_n = max_n + 1
        else:
            new_n = 1

        base_label = re.sub(r"\[Int\d+\]", "", label).rstrip()
        new_label = base_label + f" [Int{new_n}]"

        new_spec = {"label": new_label, "x": x.copy(), "y": y_int}
        self._append_or_replace_saved_spectrum(new_spec)
        self.dump_saved_spectra_to_json()

        self.ax.clear()
        self.ax.plot(x, y_int, linewidth=1.0)
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Integrated intensity (arb. units)")
        self.ax.grid(True)
        self.ax.set_title(new_label)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y_int

        self._draw_peak_markers()

    # -------- 微分 --------

    def differentiate_selected_spectrum(self):
        selection = self.spectrum_listbox.curselection()
        if len(selection) != 1:
            messagebox.showwarning("警告", "微分するスペクトルを 1 つだけ選択してください。")
            return
        idx = selection[0]
        spec = self.saved_spectra[idx]
        x = spec["x"]
        y = spec["y"]

        if len(x) < 2:
            messagebox.showwarning("警告", "データ点が少なすぎて微分できません。")
            return

        y_diff = np.gradient(y, x)

        label = spec["label"]
        diffs = re.findall(r"\[Diff(\d+)\]", label)
        if diffs:
            max_n = max(int(n) for n in diffs)
            new_n = max_n + 1
        else:
            new_n = 1

        base_label = re.sub(r"\[Diff\d+\]", "", label).rstrip()
        new_label = base_label + f" [Diff{new_n}]"

        new_spec = {"label": new_label, "x": x.copy(), "y": y_diff}
        self._append_or_replace_saved_spectrum(new_spec)
        self.dump_saved_spectra_to_json()

        self.ax.clear()
        self.ax.plot(x, y_diff, linewidth=1.0)
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("d(Intensity)/dB (arb. units)")
        self.ax.grid(True)
        self.ax.set_title(new_label)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y_diff

        self._draw_peak_markers()

    # ========= マルチピークフィット UI =========

    def on_fit_window_close(self):
        """フィットウィンドウ × ボタンで閉じるときの処理"""
        if self.fit_window is None or not self.fit_window.winfo_exists():
            return
        res = messagebox.askyesno(
            "確認",
            "現在のマルチピークフィット設定を保存しますか？"
        )
        if res:
            self._store_current_fit_meta()
        try:
            self.fit_window.destroy()
        except Exception:
            pass
        self.fit_window = None
        self.fit_target_index = None

        # 赤いピークマーカーはフィット終了で消す
        self._clear_peak_markers()

    def open_multi_peak_fit_window(self):
        selection = self.spectrum_listbox.curselection()
        if len(selection) != 1:
            messagebox.showwarning("警告", "マルチピークフィットするスペクトルを 1 つだけ選択してください。")
            return
        if not SCIPY_AVAILABLE:
            messagebox.showerror(
                "エラー",
                "scipy がインポートできないため、フィッティング機能は利用できません。\n"
                "pip install scipy を実行してから再試行してください。"
            )
            return

        # すでに他のフィットウィンドウが開いている場合は保存の有無を確認して閉じる
        if self.fit_window is not None and self.fit_window.winfo_exists():
            res = messagebox.askyesno(
                "確認",
                "別のファイルのマルチピークフィット画面を開きます。\n"
                "現在のフィット設定を保存しますか？"
            )
            if res:
                self._store_current_fit_meta()
            try:
                self.fit_window.destroy()
            except Exception:
                pass
            self.fit_window = None
            self.fit_target_index = None
            # 一旦マーカーもクリア
            self._clear_peak_markers()

        self.fit_target_index = selection[0]
        spec = self.saved_spectra[self.fit_target_index]

        win = tk.Toplevel(self)
        win.title("マルチピークフィット設定")
        win.geometry("1020x720")
        self.fit_window = win
        self.fit_window.protocol("WM_DELETE_WINDOW", self.on_fit_window_close)

        header_frame = tk.Frame(win)
        header_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        tk.Label(header_frame, text=f"対象スペクトル: {spec['label']}").pack(anchor="w")

        control_frame = tk.Frame(win)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(control_frame, text="ピーク数:").pack(side=tk.LEFT)
        self.num_peaks_entry = tk.Entry(control_frame, width=5)
        self.num_peaks_entry.insert(0, "2")
        self.num_peaks_entry.pack(side=tk.LEFT, padx=(2, 5))

        regen_btn = tk.Button(control_frame, text="行を再生成", command=self._regen_peak_rows)
        regen_btn.pack(side=tk.LEFT, padx=(0, 5))

        auto_btn = tk.Button(control_frame, text="ピーク自動推定", command=self.auto_detect_peaks)
        auto_btn.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="  検出対象:").pack(side=tk.LEFT, padx=(10, 2))
        self.peak_source_mode = tk.StringVar(value="data")
        tk.Radiobutton(control_frame, text="元データ", variable=self.peak_source_mode, value="data").pack(side=tk.LEFT)
        tk.Radiobutton(control_frame, text="一次微分", variable=self.peak_source_mode, value="deriv").pack(side=tk.LEFT)

        tk.Label(control_frame, text="  統合距離[G](全体):").pack(side=tk.LEFT, padx=(10, 2))
        self.merge_tol_entry = tk.Entry(control_frame, width=7)
        self.merge_tol_entry.insert(0, "0.5")
        self.merge_tol_entry.pack(side=tk.LEFT)

        tk.Label(control_frame, text="  表示:").pack(side=tk.LEFT, padx=(10, 2))
        self.show_individual_var = tk.BooleanVar(value=True)
        self.show_sum_var = tk.BooleanVar(value=True)
        tk.Checkbutton(control_frame, text="個別ピーク", variable=self.show_individual_var).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="合算ピーク", variable=self.show_sum_var).pack(side=tk.LEFT)

        # エリア＆グローバルパラメータ
        area_control_frame = tk.Frame(win)
        area_control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(area_control_frame, text="ピーク検出エリア数:").pack(side=tk.LEFT)
        self.num_areas_entry = tk.Entry(area_control_frame, width=5)
        self.num_areas_entry.insert(0, "0")
        self.num_areas_entry.pack(side=tk.LEFT, padx=(2, 5))

        area_regen_btn = tk.Button(area_control_frame, text="エリア行を再生成", command=self._regen_area_rows)
        area_regen_btn.pack(side=tk.LEFT, padx=(0, 10))

        tk.Label(area_control_frame, text="エリア未使用時の全体比率(0–1):").pack(side=tk.LEFT)
        self.global_peak_ratio_entry = tk.Entry(area_control_frame, width=5)
        self.global_peak_ratio_entry.insert(0, "0.1")
        self.global_peak_ratio_entry.pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(area_control_frame, text="全体・元信号最小比率(0–1):").pack(side=tk.LEFT)
        self.global_min_amp_ratio_entry = tk.Entry(area_control_frame, width=5)
        self.global_min_amp_ratio_entry.insert(0, "0.03")
        self.global_min_amp_ratio_entry.pack(side=tk.LEFT, padx=(2, 10))

        tk.Label(area_control_frame, text="全体・prom 係数(>0):").pack(side=tk.LEFT)
        self.global_prom_factor_entry = tk.Entry(area_control_frame, width=5)
        self.global_prom_factor_entry.insert(0, "0.5")
        self.global_prom_factor_entry.pack(side=tk.LEFT)

        # エリアヘッダ
        area_header = tk.Frame(win)
        area_header.pack(side=tk.TOP, fill=tk.X, padx=5)
        labels = ["使用", "Xmin[G]", "Xmax[G]", "比率(0–1)", "統合距離[G]", "元比(0–1)", "prom(>0)"]
        widths = [5, 12, 12, 10, 12, 10, 10]
        for txt, w in zip(labels, widths):
            tk.Label(area_header, text=txt, width=w).pack(side=tk.LEFT)

        self.area_rows_frame = tk.Frame(win)
        self.area_rows_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.area_rows = []

        # ピークパラメータヘッダ
        header2 = tk.Frame(win)
        header2.pack(side=tk.TOP, fill=tk.X, padx=5, pady=(8, 0))
        labels = ["使用", "関数タイプ", "中心B[G]", "FWHM[G]", "η", "強度", "fitA", "fitC", "fitW", "fitη", "×"]
        widths = [5, 15, 12, 12, 5, 10, 5, 5, 5, 5, 3]
        for txt, w in zip(labels, widths):
            tk.Label(header2, text=txt, width=w).pack(side=tk.LEFT)

        self.peak_rows_frame = tk.Frame(win)
        self.peak_rows_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        self._regen_area_rows()
        self._regen_peak_rows()
        self._apply_fit_meta_if_exists(spec)

        btn_frame = tk.Frame(win)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        preview_btn = tk.Button(btn_frame, text="推定曲線プレビュー", command=self.preview_peaks_curve)
        preview_btn.pack(side=tk.LEFT)

        fit_btn = tk.Button(btn_frame, text="フィッティング実行", command=self.run_multi_peak_fit)
        fit_btn.pack(side=tk.LEFT, padx=5)

        save_btn = tk.Button(btn_frame, text="フィット結果を保存＆表示", command=self.save_and_show_fit_results)
        save_btn.pack(side=tk.LEFT, padx=5)

    # --------- ピーク行追加（クリック用） ---------

    def _append_peak_row(self, center, amp, width, type_name="dVoigt", eta=0.0):
        """
        現在の peak_rows_frame の末尾に 1 行追加する。
        デフォルト関数タイプ: dVoigt
        """
        row = {}
        frame = tk.Frame(self.peak_rows_frame)
        frame.pack(side=tk.TOP, fill=tk.X, pady=1)

        use_var = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(frame, variable=use_var)
        cb.pack(side=tk.LEFT, padx=(0, 5))
        row["use_var"] = use_var

        type_var = tk.StringVar(value=type_name)
        type_menu = tk.OptionMenu(frame, type_var,
                                  "Gaussian", "Lorentzian", "Voigt",
                                  "dGaussian", "dLorentzian", "dVoigt")
        type_menu.config(width=10)
        type_menu.pack(side=tk.LEFT, padx=(0, 5))
        row["type_var"] = type_var

        center_entry = tk.Entry(frame, width=12)
        center_entry.insert(0, f"{center:.6g}")
        center_entry.pack(side=tk.LEFT, padx=(0, 5))
        row["center_entry"] = center_entry

        width_entry = tk.Entry(frame, width=12)
        width_entry.insert(0, f"{width:.6g}")
        width_entry.pack(side=tk.LEFT, padx=(0, 5))
        row["width_entry"] = width_entry

        eta_entry = tk.Entry(frame, width=5)
        eta_entry.insert(0, f"{eta:.6g}")
        eta_entry.pack(side=tk.LEFT, padx=(0, 5))
        row["eta_entry"] = eta_entry

        amp_entry = tk.Entry(frame, width=10)
        amp_entry.insert(0, f"{amp:.6g}")
        amp_entry.pack(side=tk.LEFT, padx=(0, 5))
        row["amp_entry"] = amp_entry

        fit_amp_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, variable=fit_amp_var).pack(side=tk.LEFT, padx=(2, 2))
        row["fit_amp_var"] = fit_amp_var

        fit_center_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, variable=fit_center_var).pack(side=tk.LEFT, padx=(2, 2))
        row["fit_center_var"] = fit_center_var

        fit_width_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, variable=fit_width_var).pack(side=tk.LEFT, padx=(2, 2))
        row["fit_width_var"] = fit_width_var

        fit_eta_var = tk.BooleanVar(value=True)
        tk.Checkbutton(frame, variable=fit_eta_var).pack(side=tk.LEFT, padx=(2, 2))
        row["fit_eta_var"] = fit_eta_var

        del_btn = tk.Button(
            frame,
            text="×",
            width=2,
            command=lambda r=row, f=frame: self._delete_peak_row(r, f)
        )
        del_btn.pack(side=tk.LEFT, padx=(4, 0))
        row["frame"] = frame
        row["del_btn"] = del_btn

        self.peak_rows.append(row)
        # ピーク数表示を更新
        self.num_peaks_entry.delete(0, tk.END)
        self.num_peaks_entry.insert(0, str(len(self.peak_rows)))

        # 行に合わせて赤線も更新
        self._update_peak_markers_from_rows()

    def _delete_peak_row(self, row, frame):
        """ピーク行を個別に削除（赤線も行に合わせて更新）"""
        if row in self.peak_rows:
            self.peak_rows.remove(row)
        try:
            frame.destroy()
        except Exception:
            pass
        # ピーク数表示を更新
        try:
            self.num_peaks_entry.delete(0, tk.END)
            self.num_peaks_entry.insert(0, str(len(self.peak_rows)))
        except Exception:
            pass

        # 行削除に応じてピークマーカー更新
        self._update_peak_markers_from_rows()

    def _regen_peak_rows(self):
        for child in getattr(self, "peak_rows_frame", []).winfo_children():
            child.destroy()
        self.peak_rows = []

        try:
            n = int(self.num_peaks_entry.get())
        except ValueError:
            n = 1
        n = max(1, min(50, n))

        for i in range(n):
            row = {}
            frame = tk.Frame(self.peak_rows_frame)
            frame.pack(side=tk.TOP, fill=tk.X, pady=1)

            use_var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(frame, variable=use_var)
            cb.pack(side=tk.LEFT, padx=(0, 5))
            row["use_var"] = use_var

            # デフォルトタイプ dVoigt
            type_var = tk.StringVar(value="dVoigt")
            type_menu = tk.OptionMenu(frame, type_var,
                                      "Gaussian", "Lorentzian", "Voigt",
                                      "dGaussian", "dLorentzian", "dVoigt")
            type_menu.config(width=10)
            type_menu.pack(side=tk.LEFT, padx=(0, 5))
            row["type_var"] = type_var

            center_entry = tk.Entry(frame, width=12)
            center_entry.insert(0, f"{(i + 1) * 100.0:.1f}")
            center_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["center_entry"] = center_entry

            width_entry = tk.Entry(frame, width=12)
            width_entry.insert(0, "10.0")
            width_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["width_entry"] = width_entry

            eta_entry = tk.Entry(frame, width=5)
            eta_entry.insert(0, "0.0")  # デフォルト η = 0.0 (純 Lorentzian)
            eta_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["eta_entry"] = eta_entry

            amp_entry = tk.Entry(frame, width=10)
            amp_entry.insert(0, "1.0")
            amp_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["amp_entry"] = amp_entry

            fit_amp_var = tk.BooleanVar(value=True)
            tk.Checkbutton(frame, variable=fit_amp_var).pack(side=tk.LEFT, padx=(2, 2))
            row["fit_amp_var"] = fit_amp_var

            fit_center_var = tk.BooleanVar(value=True)
            tk.Checkbutton(frame, variable=fit_center_var).pack(side=tk.LEFT, padx=(2, 2))
            row["fit_center_var"] = fit_center_var

            fit_width_var = tk.BooleanVar(value=True)
            tk.Checkbutton(frame, variable=fit_width_var).pack(side=tk.LEFT, padx=(2, 2))
            row["fit_width_var"] = fit_width_var

            fit_eta_var = tk.BooleanVar(value=True)
            tk.Checkbutton(frame, variable=fit_eta_var).pack(side=tk.LEFT, padx=(2, 2))
            row["fit_eta_var"] = fit_eta_var

            del_btn = tk.Button(
                frame,
                text="×",
                width=2,
                command=lambda r=row, f=frame: self._delete_peak_row(r, f)
            )
            del_btn.pack(side=tk.LEFT, padx=(4, 0))
            row["frame"] = frame
            row["del_btn"] = del_btn

            self.peak_rows.append(row)

        # 行に応じてピークマーカー更新
        self._update_peak_markers_from_rows()

    def _regen_area_rows(self):
        for child in getattr(self, "area_rows_frame", []).winfo_children():
            child.destroy()
        self.area_rows = []

        try:
            n = int(self.num_areas_entry.get())
        except ValueError:
            n = 0
        n = max(0, min(20, n))

        for i in range(n):
            row = {}
            frame = tk.Frame(self.area_rows_frame)
            frame.pack(side=tk.TOP, fill=tk.X, pady=1)

            use_var = tk.BooleanVar(value=True)
            cb = tk.Checkbutton(frame, variable=use_var)
            cb.pack(side=tk.LEFT, padx=(0, 5))
            row["use_var"] = use_var

            xmin_entry = tk.Entry(frame, width=12)
            xmin_entry.insert(0, "0.0")
            xmin_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["xmin_entry"] = xmin_entry

            xmax_entry = tk.Entry(frame, width=12)
            xmax_entry.insert(0, "0.0")
            xmax_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["xmax_entry"] = xmax_entry

            ratio_entry = tk.Entry(frame, width=10)
            ratio_entry.insert(0, "0.5")
            ratio_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["ratio_entry"] = ratio_entry

            tol_entry = tk.Entry(frame, width=12)
            tol_entry.insert(0, "0.5")  # デフォルトは 0.5 G
            tol_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["tol_entry"] = tol_entry

            min_ratio_entry = tk.Entry(frame, width=10)
            min_ratio_entry.insert(0, "")  # 空なら全体値を使用
            min_ratio_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["min_ratio_entry"] = min_ratio_entry

            prom_factor_entry = tk.Entry(frame, width=10)
            prom_factor_entry.insert(0, "")  # 空なら全体値を使用
            prom_factor_entry.pack(side=tk.LEFT, padx=(0, 5))
            row["prom_factor_entry"] = prom_factor_entry

            self.area_rows.append(row)

    def _collect_peak_params(self):
        type_map = {
            "Gaussian": 0,
            "Lorentzian": 1,
            "Voigt": 2,
            "dGaussian": 3,
            "dLorentzian": 4,
            "dVoigt": 5,
        }
        amps, centers, widths, etas, t_indices = [], [], [], [], []
        for row in self.peak_rows:
            if not row["use_var"].get():
                continue
            try:
                c = float(row["center_entry"].get())
                w = float(row["width_entry"].get())
                a = float(row["amp_entry"].get())
                eta = float(row["eta_entry"].get())
            except ValueError:
                continue
            tname = row["type_var"].get()
            t_idx = type_map.get(tname, 0)
            amps.append(a)
            centers.append(c)
            widths.append(w)
            etas.append(eta)
            t_indices.append(t_idx)
        return amps, centers, widths, etas, t_indices

    def _collect_peak_params_with_flags(self):
        type_map = {
            "Gaussian": 0,
            "Lorentzian": 1,
            "Voigt": 2,
            "dGaussian": 3,
            "dLorentzian": 4,
            "dVoigt": 5,
        }
        meta = []
        for row in self.peak_rows:
            if not row["use_var"].get():
                continue
            try:
                c = float(row["center_entry"].get())
                w = float(row["width_entry"].get())
                a = float(row["amp_entry"].get())
                eta = float(row["eta_entry"].get())
            except ValueError:
                continue
            tname = row["type_var"].get()
            t_idx = type_map.get(tname, 0)
            meta.append({
                "amp": a,
                "center": c,
                "width": w,
                "eta": eta,
                "type_idx": t_idx,
                "fit_amp": row["fit_amp_var"].get(),
                "fit_center": row["fit_center_var"].get(),
                "fit_width": row["fit_width_var"].get(),
                "fit_eta": row["fit_eta_var"].get(),
                "row": row,
            })
        return meta

    def preview_peaks_curve(self):
        if self.fit_target_index is None:
            messagebox.showwarning("警告", "まず対象スペクトルを選択してください。")
            return
        spec = self.saved_spectra[self.fit_target_index]
        x = spec["x"]
        y = spec["y"]

        amps, centers, widths, etas, t_indices = self._collect_peak_params()
        if not amps:
            messagebox.showwarning("警告", "有効なピークパラメータがありません。")
            return

        y_sum = np.zeros_like(x, dtype=float)
        components = []
        for a, c, w, eta, t_idx in zip(amps, centers, widths, etas, t_indices):
            comp = multi_peak_model(x, a, c, w, eta, t_idx)
            components.append(comp)
            y_sum += comp

        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.0, alpha=0.5, label="data")
        if self.show_sum_var.get():
            self.ax.plot(x, y_sum, linewidth=1.0, label="sum (estimate)")
        if self.show_individual_var.get():
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, comp in enumerate(components):
                self.ax.plot(x, comp, linestyle="--", linewidth=1.0,
                             color=colors[i % len(colors)],
                             label=f"pk{i+1} estimate")
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True)
        self.ax.set_title(spec["label"] + "  [peak estimate]")
        self.ax.legend(fontsize=8)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y

        # 行から赤線を再構築
        self._update_peak_markers_from_rows()

    def run_multi_peak_fit(self):
        if self.fit_target_index is None:
            messagebox.showwarning("警告", "まず対象スペクトルを選択してください。")
            return
        if not SCIPY_AVAILABLE:
            messagebox.showerror("エラー", "scipy が利用できないため、フィットは実行できません。")
            return

        spec = self.saved_spectra[self.fit_target_index]
        x = spec["x"]
        y = spec["y"]

        meta = self._collect_peak_params_with_flags()
        if not meta:
            messagebox.showwarning("警告", "有効なピークパラメータがありません。")
            return

        p0_free = []
        lb_free = []
        ub_free = []

        for m in meta:
            if m["fit_amp"]:
                p0_free.append(m["amp"])
                lb_free.append(-np.inf)
                ub_free.append(np.inf)
            if m["fit_center"]:
                p0_free.append(m["center"])
                lb_free.append(x.min())
                ub_free.append(x.max())
            if m["fit_width"]:
                p0_free.append(m["width"])
                lb_free.append(1e-6)
                ub_free.append(x.max() - x.min())
            if m["fit_eta"]:
                p0_free.append(m["eta"])
                lb_free.append(0.0)
                ub_free.append(1.0)

        if not p0_free:
            messagebox.showwarning("警告", "全てのパラメータが固定されており、フィットするものがありません。")
            return

        def model_for_fit(xdata, *p_free):
            params_full = []
            idx_free = 0
            for m in meta:
                if m["fit_amp"]:
                    amp = p_free[idx_free]; idx_free += 1
                else:
                    amp = m["amp"]
                if m["fit_center"]:
                    c = p_free[idx_free]; idx_free += 1
                else:
                    c = m["center"]
                if m["fit_width"]:
                    w = p_free[idx_free]; idx_free += 1
                else:
                    w = m["width"]
                if m["fit_eta"]:
                    eta = p_free[idx_free]; idx_free += 1
                else:
                    eta = m["eta"]
                params_full.extend([amp, c, w, eta, m["type_idx"]])
            return multi_peak_model(xdata, *params_full)

        try:
            popt, pcov = curve_fit(
                model_for_fit, x, y,
                p0=p0_free,
                bounds=(lb_free, ub_free),
                maxfev=20000,
            )
        except Exception as e:
            messagebox.showerror("エラー", f"フィットに失敗しました:\n{e}")
            return

        idx_free = 0
        type_rev = {0: "Gaussian", 1: "Lorentzian", 2: "Voigt",
                    3: "dGaussian", 4: "dLorentzian", 5: "dVoigt"}
        for m in meta:
            row = m["row"]
            amp = m["amp"]
            c = m["center"]
            w = m["width"]
            eta = m["eta"]
            if m["fit_amp"]:
                amp = popt[idx_free]; idx_free += 1
            if m["fit_center"]:
                c = popt[idx_free]; idx_free += 1
            if m["fit_width"]:
                w = popt[idx_free]; idx_free += 1
            if m["fit_eta"]:
                eta = popt[idx_free]; idx_free += 1

            row["amp_entry"].delete(0, tk.END)
            row["amp_entry"].insert(0, f"{amp:.6g}")
            row["center_entry"].delete(0, tk.END)
            row["center_entry"].insert(0, f"{c:.6g}")
            row["width_entry"].delete(0, tk.END)
            row["width_entry"].insert(0, f"{w:.6g}")
            row["eta_entry"].delete(0, tk.END)
            row["eta_entry"].insert(0, f"{eta:.6g}")
            row["type_var"].set(type_rev.get(m["type_idx"], "Gaussian"))

        self.preview_peaks_curve()
        self._store_current_fit_meta()

    def save_and_show_fit_results(self):
        if self.fit_target_index is None:
            messagebox.showwarning("警告", "まず対象スペクトルを選択してください。")
            return

        spec = self.saved_spectra[self.fit_target_index]
        x = spec["x"]
        y = spec["y"]
        base_label = spec["label"]

        amps, centers, widths, etas, t_indices = self._collect_peak_params()
        if not amps:
            messagebox.showwarning("警告", "有効なピークパラメータがありません。")
            return

        components = []
        for a, c, w, eta, t_idx in zip(amps, centers, widths, etas, t_indices):
            comp = multi_peak_model(x, a, c, w, eta, t_idx)
            components.append(comp)
        y_sum = np.sum(components, axis=0)

        # ★ 同じラベルがあれば上書き保存する
        for i, comp in enumerate(components):
            label = base_label + f" [Fit pk{i+1}]"
            spec_i = {"label": label, "x": x.copy(), "y": comp.copy()}
            self._append_or_replace_saved_spectrum(spec_i)

        sum_label = base_label + " [Fit sum]"
        spec_sum = {"label": sum_label, "x": x.copy(), "y": y_sum.copy()}
        self._append_or_replace_saved_spectrum(spec_sum)

        self.dump_saved_spectra_to_json()

        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.0, alpha=0.4, label="data")
        if self.show_sum_var.get():
            self.ax.plot(x, y_sum, linewidth=1.5, label="fit sum")
        if self.show_individual_var.get():
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            for i, comp in enumerate(components):
                self.ax.plot(x, comp, linestyle="--", linewidth=1.0,
                             color=colors[i % len(colors)], label=f"pk{i+1}")
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid(True)
        self.ax.set_title(base_label + "  [multi-peak fit]")
        self.ax.legend(fontsize=8)
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y

        # 行から赤線を再構築
        self._update_peak_markers_from_rows()

    # -------- メインプロットにピーク線を描画 --------

    def show_peak_markers(self, x, y, centers):
        centers = np.asarray(centers, dtype=float)
        self.ax.clear()
        self.ax.plot(x, y, linewidth=1.0)
        self.ax.set_xlabel("Magnetic field (G, corrected)")
        self.ax.set_ylabel("Intensity (normalized arb. units)")
        self.ax.grid(True)
        self.ax.set_title("ピーク位置マーカー付き")
        self.canvas.draw()

        self.pointer_source_x = x
        self.pointer_source_y = y

        self.peak_centers = list(centers)
        self._draw_peak_markers()

    def _draw_peak_markers(self):
        """
        self.peak_centers に基づいて短い赤線を描画
        （既存マーカーは self.peak_marker_artists から削除）
        """

        # 既存ピークマーカーを削除
        for art in self.peak_marker_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.peak_marker_artists = []

        if not self.peak_centers:
            self.canvas.draw()
            return

        y_min, y_max = self.ax.get_ylim()
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            self.canvas.draw()
            return

        yr = y_max - y_min if (y_max - y_min) > 0 else 1.0
        mid = (y_min + y_max) * 0.5
        half_len = 0.08 * yr  # 短い線

        for c in self.peak_centers:
            art = self.ax.vlines(
                c,
                mid - half_len,
                mid + half_len,
                colors="red",
                linestyles="--",
                linewidth=1.0
            )
            self.peak_marker_artists.append(art)

        self.canvas.draw()

    def _clear_peak_markers(self):
        """
        peak_centers をクリアし、ピークマーカー線だけを remove
        """
        self.peak_centers = []

        for art in self.peak_marker_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.peak_marker_artists = []

        self.canvas.draw()




    def _update_peak_markers_from_rows(self):
        """peak_rows の内容に基づいて peak_centers を再構成し、赤線を描き直す"""
        centers = []
        for row in getattr(self, "peak_rows", []):
            if not row["use_var"].get():
                continue
            try:
                c = float(row["center_entry"].get())
            except Exception:
                continue
            centers.append(c)
        self.peak_centers = centers
        self._draw_peak_markers()

    # -------- 近いピークを統合するヘルパー --------

    def _merge_peaks_by_tolerance(self, centers, base_indices, tol, amps=None):
        """
        近いピークを統合するヘルパー関数
        centers: ndarray (ピーク中心の x)
        base_indices: ndarray or None（元配列のインデックス）
        tol: [G] この値以下で隣接するピークは同じクラスタとしてまとめる
        amps: ndarray or None (統合代表を選ぶための |振幅| 等)
        戻り値: (merged_centers, merged_indices or None)
        """
        centers = np.asarray(centers, dtype=float)
        if base_indices is not None:
            base_indices = np.asarray(base_indices, dtype=int)
        if amps is not None:
            amps = np.asarray(amps, dtype=float)

        if len(centers) == 0:
            return centers, base_indices

        order = np.argsort(centers)
        centers = centers[order]
        if base_indices is not None:
            base_indices = base_indices[order]
        if amps is not None:
            amps = amps[order]

        merged_centers = []
        merged_indices = [] if base_indices is not None else None

        current_group = [0]
        for i in range(1, len(centers)):
            if abs(centers[i] - centers[current_group[-1]]) <= tol:
                current_group.append(i)
            else:
                g = np.array(current_group)
                if amps is not None:
                    idx_local = np.argmax(amps[g])
                else:
                    idx_local = np.argmin(np.abs(centers[g] - np.mean(centers[g])))
                idx_global = g[idx_local]
                merged_centers.append(float(centers[idx_global]))
                if base_indices is not None:
                    merged_indices.append(int(base_indices[idx_global]))
                current_group = [i]

        g = np.array(current_group)
        if amps is not None:
            idx_local = np.argmax(amps[g])
        else:
            idx_local = np.argmin(np.abs(centers[g] - np.mean(centers[g])))
        idx_global = g[idx_local]
        merged_centers.append(float(centers[idx_global]))
        if base_indices is not None:
            merged_indices.append(int(base_indices[idx_global]))

        merged_centers = np.array(merged_centers, dtype=float)
        if base_indices is not None:
            merged_indices = np.array(merged_indices, dtype=int)
        else:
            merged_indices = None
        return merged_centers, merged_indices

    # -------- フィット設定の保存・復元 --------

    def _store_current_fit_meta(self):
        if self.fit_target_index is None:
            return
        spec = self.saved_spectra[self.fit_target_index]

        meta = {}
        try:
            meta["peak_source_mode"] = self.peak_source_mode.get()
        except Exception:
            meta["peak_source_mode"] = "data"

        try:
            meta["merge_tol_global"] = float(self.merge_tol_entry.get())
        except Exception:
            meta["merge_tol_global"] = 0.5

        try:
            meta["global_peak_ratio"] = float(self.global_peak_ratio_entry.get())
        except Exception:
            meta["global_peak_ratio"] = 0.1

        try:
            meta["global_min_amp_ratio"] = float(self.global_min_amp_ratio_entry.get())
        except Exception:
            meta["global_min_amp_ratio"] = 0.03

        try:
            meta["global_prom_factor"] = float(self.global_prom_factor_entry.get())
        except Exception:
            meta["global_prom_factor"] = 0.5

        areas = []
        for row in self.area_rows:
            try:
                use = bool(row["use_var"].get())
                xmin = float(row["xmin_entry"].get())
                xmax = float(row["xmax_entry"].get())
                ratio = float(row["ratio_entry"].get())
                tol = float(row["tol_entry"].get())
                min_ratio_txt = row["min_ratio_entry"].get()
                prom_txt = row["prom_factor_entry"].get()
                min_ratio = float(min_ratio_txt) if min_ratio_txt.strip() != "" else None
                prom_factor = float(prom_txt) if prom_txt.strip() != "" else None
            except Exception:
                continue
            areas.append({
                "use": use,
                "xmin": xmin,
                "xmax": xmax,
                "ratio": ratio,
                "merge_tol": tol,
                "min_ratio": min_ratio,
                "prom_factor": prom_factor,
            })
        meta["areas"] = areas
        meta["num_areas"] = len(areas)

        peaks = []
        for row in self.peak_rows:
            try:
                use = bool(row["use_var"].get())
                c = float(row["center_entry"].get())
                w = float(row["width_entry"].get())
                a = float(row["amp_entry"].get())
                eta = float(row["eta_entry"].get())
            except ValueError:
                continue
            peaks.append({
                "use": use,
                "type": row["type_var"].get(),
                "center": c,
                "width": w,
                "eta": eta,
                "amp": a,
                "fit_amp": bool(row["fit_amp_var"].get()),
                "fit_center": bool(row["fit_center_var"].get()),
                "fit_width": bool(row["fit_width_var"].get()),
                "fit_eta": bool(row["fit_eta_var"].get()),
            })
        meta["peaks"] = peaks
        meta["num_peaks"] = len(peaks)

        spec["fit_meta"] = meta
        self.saved_spectra[self.fit_target_index] = spec
        self.dump_saved_spectra_to_json()

    def _apply_fit_meta_if_exists(self, spec):
        meta = spec.get("fit_meta")
        if not meta:
            return

        try:
            self.peak_source_mode.set(meta.get("peak_source_mode", "data"))
        except Exception:
            pass

        try:
            mt = float(meta.get("merge_tol_global", 0.5))
            self.merge_tol_entry.delete(0, tk.END)
            self.merge_tol_entry.insert(0, f"{mt:.3g}")
        except Exception:
            pass

        try:
            gr = float(meta.get("global_peak_ratio", 0.1))
            self.global_peak_ratio_entry.delete(0, tk.END)
            self.global_peak_ratio_entry.insert(0, f"{gr:.3g}")
        except Exception:
            pass

        try:
            gm = float(meta.get("global_min_amp_ratio", 0.03))
            self.global_min_amp_ratio_entry.delete(0, tk.END)
            self.global_min_amp_ratio_entry.insert(0, f"{gm:.3g}")
        except Exception:
            pass

        try:
            gf = float(meta.get("global_prom_factor", 0.5))
            self.global_prom_factor_entry.delete(0, tk.END)
            self.global_prom_factor_entry.insert(0, f"{gf:.3g}")
        except Exception:
            pass

        areas = meta.get("areas", [])
        try:
            self.num_areas_entry.delete(0, tk.END)
            self.num_areas_entry.insert(0, str(len(areas)))
            self._regen_area_rows()
            for row, a in zip(self.area_rows, areas):
                row["use_var"].set(bool(a.get("use", True)))
                row["xmin_entry"].delete(0, tk.END)
                row["xmin_entry"].insert(0, f"{a.get('xmin', 0.0):.6g}")
                row["xmax_entry"].delete(0, tk.END)
                row["xmax_entry"].insert(0, f"{a.get('xmax', 0.0):.6g}")
                row["ratio_entry"].delete(0, tk.END)
                row["ratio_entry"].insert(0, f"{a.get('ratio', 0.5):.6g}")
                row["tol_entry"].delete(0, tk.END)
                row["tol_entry"].insert(0, f"{a.get('merge_tol', 0.5):.6g}")

                min_ratio = a.get("min_ratio", None)
                prom_factor = a.get("prom_factor", None)

                row["min_ratio_entry"].delete(0, tk.END)
                if min_ratio is not None:
                    row["min_ratio_entry"].insert(0, f"{min_ratio:.6g}")

                row["prom_factor_entry"].delete(0, tk.END)
                if prom_factor is not None:
                    row["prom_factor_entry"].insert(0, f"{prom_factor:.6g}")
        except Exception:
            pass

        peaks = meta.get("peaks", [])
        try:
            self.num_peaks_entry.delete(0, tk.END)
            self.num_peaks_entry.insert(0, str(len(peaks)))
            self._regen_peak_rows()
            for row, p in zip(self.peak_rows, peaks):
                row["use_var"].set(bool(p.get("use", True)))
                row["type_var"].set(p.get("type", "dVoigt"))
                row["center_entry"].delete(0, tk.END)
                row["center_entry"].insert(0, f"{p.get('center', 0.0):.6g}")
                row["width_entry"].delete(0, tk.END)
                row["width_entry"].insert(0, f"{p.get('width', 1.0):.6g}")
                row["eta_entry"].delete(0, tk.END)
                row["eta_entry"].insert(0, f"{p.get('eta', 0.0):.6g}")
                row["amp_entry"].delete(0, tk.END)
                row["amp_entry"].insert(0, f"{p.get('amp', 1.0):.6g}")
                row["fit_amp_var"].set(bool(p.get("fit_amp", True)))
                row["fit_center_var"].set(bool(p.get("fit_center", True)))
                row["fit_width_var"].set(bool(p.get("fit_width", True)))
                row["fit_eta_var"].set(bool(p.get("fit_eta", True)))
        except Exception:
            pass

        # メタ適用後に赤線更新
        self._update_peak_markers_from_rows()

    # -------- 自動ピーク検出（元データ / 一次微分） --------

    def auto_detect_peaks(self):
        """
        ピーク自動推定
        - 検出対象: 元データ  → find_peaks で極大・極小（正負）
        - 検出対象: 一次微分 → y を平滑化してから微分し、そのピークを検出
          + 元信号の |y| が全体最大値の (元比) 以上の位置のみ採用（元比は全体 or エリアごとに設定）
          + prominence は height * prom係数（全体 or エリアごと）
        """
        if self.fit_target_index is None:
            messagebox.showwarning("警告", "まず対象スペクトルを選択してください。")
            return
        if not SCIPY_AVAILABLE:
            messagebox.showerror("エラー", "scipy が利用できないため、自動ピーク検出は利用できません。")
            return

        spec = self.saved_spectra[self.fit_target_index]
        x = spec["x"]
        y = spec["y"]

        mode = self.peak_source_mode.get()  # "data" or "deriv"

        try:
            merge_tol_global = float(self.merge_tol_entry.get())
        except Exception:
            merge_tol_global = 0.0
        if merge_tol_global < 0:
            merge_tol_global = 0.0

        # グローバル設定
        try:
            global_min_ratio = float(self.global_min_amp_ratio_entry.get())
        except Exception:
            global_min_ratio = 0.03
        global_min_ratio = max(0.0, min(1.0, global_min_ratio))

        try:
            global_prom_factor = float(self.global_prom_factor_entry.get())
        except Exception:
            global_prom_factor = 0.5
        if global_prom_factor <= 0:
            global_prom_factor = 0.5

        # エリア: (xmin, xmax, ratio, tol_area, min_ratio, prom_factor)
        valid_areas = []
        for row in self.area_rows:
            try:
                if not row["use_var"].get():
                    continue
                xmin = float(row["xmin_entry"].get())
                xmax = float(row["xmax_entry"].get())
                ratio = float(row["ratio_entry"].get())
                tol_area = float(row["tol_entry"].get())
                min_ratio_txt = row["min_ratio_entry"].get()
                prom_txt = row["prom_factor_entry"].get()
                min_ratio = float(min_ratio_txt) if min_ratio_txt.strip() != "" else None
                prom_factor = float(prom_txt) if prom_txt.strip() != "" else None
            except (ValueError, tk.TclError):
                continue
            if xmax <= xmin:
                continue
            ratio = max(0.0, min(1.0, ratio))
            if tol_area < 0:
                tol_area = 0.0
            if min_ratio is not None:
                min_ratio = max(0.0, min(1.0, min_ratio))
            if prom_factor is not None and prom_factor <= 0:
                prom_factor = None
            valid_areas.append((xmin, xmax, ratio, tol_area, min_ratio, prom_factor))

        # ============================================================
        # 1) 元データモード
        # ============================================================
        if mode == "data":
            indices_all = []

            def detect_in_range_data(xmin, xmax, ratio):
                mask = (x >= xmin) & (x <= xmax)
                idx_range = np.where(mask)[0]
                if len(idx_range) < 3:
                    return np.array([], dtype=int)

                sig_area = y[idx_range]
                max_abs = float(np.max(np.abs(sig_area)))
                if max_abs <= 0:
                    return np.array([], dtype=int)

                ratio_clamped = max(0.0, min(1.0, ratio))
                height_thr = ratio_clamped * max_abs

                idx_pos, _ = find_peaks(sig_area, height=height_thr)
                idx_neg, _ = find_peaks(-sig_area, height=height_thr)

                result = []
                if len(idx_pos) > 0:
                    result.extend(idx_range[idx_pos])
                if len(idx_neg) > 0:
                    result.extend(idx_range[idx_neg])
                if not result:
                    return np.array([], dtype=int)
                return np.unique(np.array(result, dtype=int))

            if valid_areas:
                for xmin, xmax, ratio, tol_area, min_ratio, prom_factor in valid_areas:
                    idxs = detect_in_range_data(xmin, xmax, ratio)
                    if idxs.size == 0:
                        continue
                    # 統合距離によるマージ（元データの場合は amps = |y|）
                    if tol_area > 0.0 and idxs.size > 1:
                        centers = x[idxs]
                        amps = np.abs(y[idxs])
                        _, merged_indices = self._merge_peaks_by_tolerance(
                            centers, idxs, tol_area, amps
                        )
                        idxs = merged_indices
                    indices_all.extend(list(idxs))
            else:
                try:
                    ratio = float(self.global_peak_ratio_entry.get())
                except ValueError:
                    ratio = 0.1
                ratio = max(0.0, min(1.0, ratio))
                idxs = detect_in_range_data(x.min(), x.max(), ratio)
                if idxs.size > 1 and merge_tol_global > 0.0:
                    centers = x[idxs]
                    amps = np.abs(y[idxs])
                    _, merged_indices = self._merge_peaks_by_tolerance(
                        centers, idxs, merge_tol_global, amps
                    )
                    idxs = merged_indices
                indices_all.extend(list(idxs))

            if not indices_all:
                messagebox.showwarning("警告", "ピークが検出できませんでした。エリアや比率、モードを調整してみてください。")
                return

            indices = np.unique(np.array(indices_all, dtype=int))
            indices = np.sort(indices)

            n_peaks = len(indices)
            self.num_peaks_entry.delete(0, tk.END)
            self.num_peaks_entry.insert(0, str(n_peaks))
            self._regen_peak_rows()

            width_init = (x.max() - x.min()) / 50.0

            centers_list = []
            for row, idx in zip(self.peak_rows, indices):
                xc = x[idx]
                centers_list.append(xc)

                row["center_entry"].delete(0, tk.END)
                row["center_entry"].insert(0, f"{xc:.6g}")

                amp0 = float(y[idx])
                row["amp_entry"].delete(0, tk.END)
                row["amp_entry"].insert(0, f"{amp0:.6g}")

                row["width_entry"].delete(0, tk.END)
                row["width_entry"].insert(0, f"{width_init:.6g}")

                row["eta_entry"].delete(0, tk.END)
                row["eta_entry"].insert(0, "0.0")  # デフォルト

                # デフォルト関数タイプ dVoigt
                row["type_var"].set("dVoigt")
                row["use_var"].set(True)

            # マーカー付きで表示
            self.show_peak_markers(x, y, centers_list)

            messagebox.showinfo("自動推定", f"{n_peaks} 個のピークを自動検出し、GUI に反映しました。")
            return

        # ============================================================
        # 2) 一次微分モード：
        #    y を平滑化してから微分し、そのピークを検出（ノイズ抑制）
        # ============================================================
        elif mode == "deriv":
            # まず y を移動平均で平滑化
            n = len(x)
            win_s = max(3, n // 200)
            if win_s % 2 == 0:
                win_s += 1
            kernel = np.ones(win_s) / win_s
            y_smooth = np.convolve(y, kernel, mode="same")

            # 平滑化した y から 1次微分
            y1 = np.gradient(y_smooth, x)

            centers_all = []
            amp_indices_all = []

            global_abs_max = float(np.max(np.abs(y))) if np.max(np.abs(y)) > 0 else 1.0

            def detect_in_range_deriv(xmin, xmax, ratio, tol_area, min_ratio, prom_factor):
                # ローカル設定（未指定ならグローバル）
                local_min_ratio = global_min_ratio if min_ratio is None else min_ratio
                local_prom_factor = global_prom_factor if prom_factor is None else prom_factor

                mask = (x >= xmin) & (x <= xmax)
                idx_range = np.where(mask)[0]
                if len(idx_range) < 3:
                    return [], []

                y_area = y1[idx_range]
                max_abs = float(np.max(np.abs(y_area)))
                if max_abs <= 0:
                    return [], []

                ratio_clamped = max(0.0, min(1.0, ratio))
                thr = ratio_clamped * max_abs

                # height と prominence でノイズ抑制
                peak_idx, _ = find_peaks(
                    np.abs(y_area),
                    height=thr,
                    prominence=max(thr * local_prom_factor, 0.0)
                )
                if len(peak_idx) == 0:
                    return [], []

                idxs = idx_range[peak_idx]

                # エリア内の統合距離でマージ（2次微分ピーク高さで代表を選択）
                if tol_area > 0.0 and len(idxs) > 1:
                    centers = x[idxs]
                    amps_local = np.abs(y1[idxs])
                    merged_centers, merged_indices = self._merge_peaks_by_tolerance(
                        centers, idxs, tol_area, amps_local
                    )
                    idxs = merged_indices

                # 元信号の振幅が小さすぎるもの（ノイズ）は除外
                filtered_centers = []
                filtered_indices = []
                for idx in idxs:
                    if abs(y[idx]) >= local_min_ratio * global_abs_max:
                        filtered_centers.append(x[idx])
                        filtered_indices.append(idx)

                return filtered_centers, filtered_indices

            if valid_areas:
                for xmin, xmax, ratio, tol_area, min_ratio, prom_factor in valid_areas:
                    c_list, idx_list = detect_in_range_deriv(xmin, xmax, ratio, tol_area, min_ratio, prom_factor)
                    centers_all.extend(c_list)
                    amp_indices_all.extend(idx_list)
            else:
                try:
                    ratio = float(self.global_peak_ratio_entry.get())
                except ValueError:
                    ratio = 0.1
                ratio = max(0.0, min(1.0, ratio))

                c_list, idx_list = detect_in_range_deriv(
                    x.min(), x.max(), ratio, merge_tol_global, None, None
                )
                centers_all.extend(c_list)
                amp_indices_all.extend(idx_list)

            if not centers_all:
                messagebox.showwarning("警告", "ピークが検出できませんでした。エリアや比率、元比・prom係数、モードを調整してみてください。")
                return

            centers_all = np.array(centers_all, dtype=float)
            amp_indices_all = np.array(amp_indices_all, dtype=int)
            order = np.argsort(centers_all)
            centers_all = centers_all[order]
            amp_indices_all = amp_indices_all[order]

            n_peaks = len(centers_all)
            self.num_peaks_entry.delete(0, tk.END)
            self.num_peaks_entry.insert(0, str(n_peaks))
            self._regen_peak_rows()

            width_init = (x.max() - x.min()) / 50.0

            for row, x0, idx_amp in zip(self.peak_rows, centers_all, amp_indices_all):
                row["center_entry"].delete(0, tk.END)
                row["center_entry"].insert(0, f"{x0:.6g}")

                amp0 = float(y[idx_amp])  # 初期強度は元の y
                row["amp_entry"].delete(0, tk.END)
                row["amp_entry"].insert(0, f"{amp0:.6g}")

                row["width_entry"].delete(0, tk.END)
                row["width_entry"].insert(0, f"{width_init:.6g}")

                row["eta_entry"].delete(0, tk.END)
                row["eta_entry"].insert(0, "0.0")  # デフォルト

                # デフォルト関数タイプ dVoigt
                row["type_var"].set("dVoigt")
                row["use_var"].set(True)

            # マーカー付きで表示
            self.show_peak_markers(x, y, centers_all)

            messagebox.showinfo("自動推定", f"{n_peaks} 個のピークを自動検出し、GUI に反映しました。")

    # ===== 計測カーソル関連 =====

    def set_cursor_mode(self, mode):
        if mode not in ("A", "B"):
            self.cursor_mode = None
            self.cursor_mode_label.config(text="モード: なし")
            return
        self.cursor_mode = mode
        self.cursor_mode_label.config(text=f"モード: カーソル{mode}セット")

    def clear_cursors(self):
        self.cursor_mode = None
        self.cursor_mode_label.config(text="モード: なし")
        self.cursorA = None
        self.cursorB = None
        for art in self.cursorA_artists + self.cursorB_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.cursorA_artists = []
        self.cursorB_artists = []
        self.cursor_info_var.set("カーソルA/B: 未設定")
        self.canvas.draw()

    def _set_cursor_point_from_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return

        # pointer_source があれば最近傍点を使う
        if self.pointer_source_x is not None and self.pointer_source_y is not None:
            x_arr = self.pointer_source_x
            y_arr = self.pointer_source_y
            idx = int(np.argmin(np.abs(x_arr - event.xdata)))
            x_val = float(x_arr[idx])
            y_val = float(y_arr[idx])
        else:
            x_val = float(event.xdata)
            y_val = float(event.ydata) if event.ydata is not None else 0.0

        if self.cursor_mode == "A":
            self.cursorA = (x_val, y_val)
        elif self.cursor_mode == "B":
            self.cursorB = (x_val, y_val)

        self._draw_cursor_markers()
        self._update_cursor_info()

    def _draw_cursor_markers(self):
        # 既存カーソルマーカーを消去
        for art in self.cursorA_artists + self.cursorB_artists:
            try:
                art.remove()
            except Exception:
                pass
        self.cursorA_artists = []
        self.cursorB_artists = []

        y_min, y_max = self.ax.get_ylim()
        yr = y_max - y_min if (y_max - y_min) > 0 else 1.0

        if self.cursorA is not None:
            xa, ya = self.cursorA
            v = self.ax.vlines(xa, ya - 0.04 * yr, ya + 0.04 * yr)
            if self.pointer_source_x is not None and len(self.pointer_source_x) > 1:
                xr = self.pointer_source_x
                xspan = float(xr.max() - xr.min())
                hx = 0.04 * xspan
            else:
                hx = 0.04
            h = self.ax.hlines(ya, xa - hx, xa + hx)
            p = self.ax.plot(xa, ya, marker="x", markersize=8)[0]
            self.cursorA_artists.extend([v, h, p])

        if self.cursorB is not None:
            xb, yb = self.cursorB
            v = self.ax.vlines(xb, yb - 0.04 * yr, yb + 0.04 * yr)
            if self.pointer_source_x is not None and len(self.pointer_source_x) > 1:
                xr = self.pointer_source_x
                xspan = float(xr.max() - xr.min())
                hx = 0.04 * xspan
            else:
                hx = 0.04
            h = self.ax.hlines(yb, xb - hx, xb + hx)
            p = self.ax.plot(xb, yb, marker="+", markersize=8)[0]
            self.cursorB_artists.extend([v, h, p])

        self.canvas.draw()

    def _update_cursor_info(self):
        if self.cursorA is None and self.cursorB is None:
            self.cursor_info_var.set("カーソルA/B: 未設定")
            return
        text = []
        if self.cursorA is not None:
            xa, ya = self.cursorA
            text.append(f"A: X={xa:.4f} G, Y={ya:.4g}")
        else:
            text.append("A: -")
        if self.cursorB is not None:
            xb, yb = self.cursorB
            text.append(f"B: X={xb:.4f} G, Y={yb:.4g}")
        else:
            text.append("B: -")
        if self.cursorA is not None and self.cursorB is not None:
            dx = self.cursorB[0] - self.cursorA[0]
            dy = self.cursorB[1] - self.cursorA[1]
            text.append(f"ΔX=B−A={dx:.4f} G, ΔY={dy:.4g}")
        self.cursor_info_var.set(" | ".join(text))


# -------- メイン --------

if __name__ == "__main__":
    app = EPRViewerApp()
    app.mainloop()
