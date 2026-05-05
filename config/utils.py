@property
def OUTPUT_FILE(self):
    iir_str = f"IIR-{self.L_FREQ}-{self.H_FREQ}" if self.L_FREQ is not None else "IIR-False"
    res_str = f"Res-{self.FS_RESAMPLE}" if self.FS_RESAMPLE is not None else "Res-False"
    reg_str = f"Reg-{self.CSP_REGULARIZATION}" if hasattr(self, 'CSP_REGULARIZATION') else "Reg-None"

    if self.REFERENCE_CHANNELS == 'average':
        ref_str = "Ref-CAR"
    elif isinstance(self.REFERENCE_CHANNELS, list) and len(self.REFERENCE_CHANNELS) > 0:
        ref_str = f"Ref-{self.REFERENCE_CHANNELS[0]}" # تبدیل می‌شود به Ref-Cz
    else:
        ref_str = "Ref-None"

    filename = f"features_{self.EXTRACTOR_TYPE}_{iir_str}_{ref_str}_{res_str}_{reg_str}.npz"
    return f"data/processed/{filename}"