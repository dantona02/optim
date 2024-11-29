from pathlib import Path
import numpy as np
from tqdm import tqdm
from typing import Tuple, Union
from bmc.bmc_tool import BMCTool
from bmc.bmc_tool import prep_rf_simulation
from bmc.params import Params


class FID(BMCTool):
    def __init__(self, adc_time: np.float64, params: Params, seq_file: str | Path, verbose: bool = True, **kwargs) -> None:
        super().__init__(params, seq_file, verbose, **kwargs)
        """
        Parameters
        ----------
        adc_time: np.float64
            readout duration in seconds
        params : Params
            Params object containing all simulation parameters
        seq_file : Union[str, Path]
            Path to the seq-file
        verbose : bool, optional
            Flag to activate detailed outpus, by default True
        """
        self.adc_time = adc_time
        self.defs["num_meas"] = self.params.options["max_pulse_samples"]

        if "num_meas" in self.defs:
            self.n_measure = int(self.defs["num_meas"]) #redefining n_measure to max_pulse_samples
        else:
            self.n_measure = self.n_offsets
            
        self.m_out = np.zeros([self.m_init.shape[0], self.n_measure]) #expanding m_out to the number of max_pulse_samples
        self.dt = self.adc_time / self.params.options["max_pulse_samples"]


    def run_adc(self, block, current_adc, accum_phase, mag) -> Tuple[int, float, np.ndarray]:

        #adc with time dt and max_pulse_sampels
        if block.adc is not None:
            
            for step in range(self.params.options["max_pulse_samples"]):
                self.m_out[:, current_adc] = np.squeeze(mag)
                
                accum_phase = 0
                current_adc += 1

                self.bm_solver.update_matrix(0, 0, 0) #no rf_amp, no rf_phase, no rf_freq
                mag = self.bm_solver.solve_equation(mag=mag, dtp=self.dt)
            
            # RF pulse
        elif block.rf is not None:
            amp_, ph_, dtp_, delay_after_pulse = prep_rf_simulation(block, self.params.options["max_pulse_samples"])
            
            for i in range(amp_.size):
                self.bm_solver.update_matrix(
                    rf_amp=amp_[i],
                    rf_phase=-ph_[i] + block.rf.phase_offset - accum_phase,
                    rf_freq=block.rf.freq_offset,
                )
                # print(np.squeeze(mag))
                mag = self.bm_solver.solve_equation(mag=mag, dtp=dtp_)
                

            if delay_after_pulse > 0:
                self.bm_solver.update_matrix(0, 0, 0)
                mag = self.bm_solver.solve_equation(mag=mag, dtp=delay_after_pulse)

            phase_degree = dtp_ * amp_.size * 360 * block.rf.freq_offset
            phase_degree %= 360
            accum_phase += phase_degree / 180 * np.pi

        return current_adc, accum_phase, mag

    def run_fid(self) -> None:
        """
        Start simulation process.
        """
        if self.n_offsets != self.n_measure:
            self.run_m0_scan = True

        current_adc = 0
        accum_phase = 0
        mag = self.m_init[np.newaxis, :, np.newaxis]

        try:
            block_events = self.seq.block_events
        except AttributeError:
            block_events = self.seq.dict_block_events

        if self.verbose:
            loop_block_events = tqdm(range(1, len(block_events) + 1), desc="BMCTool simulation")
        else:
            loop_block_events = range(1, len(block_events) + 1)

        # code for pypulseq >= 1.4.0:
        try:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_adc(block, current_adc, accum_phase, mag)
        except AttributeError:
            for block_event in loop_block_events:
                block = self.seq.get_block(block_event)
                current_adc, accum_phase, mag = self.run_1_3_0(block, current_adc, accum_phase, mag)

    def get_magtrans(self, return_zmag: bool = False, return_cest_pool: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rturns the complex transverse magnetization of water pool or one cest pool. No implementation for MT pools.
        If return_cest_pool is True, only returns magnetization of the first cest pool.
        ----------
        Parameters
        ----------
        return_zmag: bool, optional
            If True, returns z-magnetization
        return_cest_pool: bool, optional
            If True, returns magnetization of the first cest pool
        """

        t = np.arange(0, self.adc_time, self.dt)
        if return_cest_pool and self.params.cest_pools:
            if return_zmag:
                m_z = self.m_out[self.params.mz_loc + 1, :]
                return t, np.abs(m_z)
            else:
                n_total_pools = len(self.params.cest_pools) + 1
                m_trans_c = self.m_out[1, :] + 1j * self.m_out[n_total_pools + 1, :]
        else:
            if return_zmag:
                m_z = self.m_out[self.params.mz_loc, :]
                return t, m_z
            elif self.params.cest_pools:
                n_total_pools = len(self.params.cest_pools) + 1
                m_trans_c = self.m_out[0, :] + 1j * self.m_out[n_total_pools, :]
            else:
                m_trans_c = self.m_out[0, :] + 1j * self.m_out[1, :]
                return t, m_trans_c
    