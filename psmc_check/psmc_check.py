#!/usr/bin/env python

"""
========================
psmc_check
========================

This code generates backstop load review outputs for checking the ACIS
PSMC temperature 1PDEAAT.  It also generates PSMC model validation
plots comparing predicted values to telemetry for the previous three
weeks.
"""
from __future__ import print_function

# Matplotlib setup                                                                                                                                              
# Use Agg backend for command-line (non-interactive) operation                                                                                                   
import matplotlib
matplotlib.use('Agg')

import logging
import Chandra.cmd_states as cmd_states
from astropy.io import ascii
from Chandra.Time import DateTime, date2secs
import numpy as np
import xija
from acis_thermal_check.main import ACISThermalCheck
from acis_thermal_check.utils import calc_off_nom_rolls, get_options
import os
import sys

model_path = os.path.abspath(os.path.dirname(__file__))

MSID = dict(psmc='1PDEAAT')
YELLOW = dict(psmc=55.0)
MARGIN = dict(psmc=2.5)
VALIDATION_LIMITS = {'1PDEAAT': [(1, 2.5),
                                 (50, 1.0),
                                 (99, 5.5)],
                     'PITCH': [(1, 3.0),
                               (99, 3.0)],
                     'TSCPOS': [(1, 2.5),
                                (99, 2.5)]
                     }
HIST_LIMIT = [30., 40.]

logger = logging.getLogger('psmc_check')

def calc_model(model_spec, states, start, stop, T_psmc=None, T_psmc_times=None,
               T_pin1at=None,T_pin1at_times=None,
               dh_heater=None,dh_heater_times=None):
    model = xija.XijaModel('psmc', start=start, stop=stop, model_spec=model_spec)
    times = np.array([states['tstart'], states['tstop']])
    model.comp['sim_z'].set_data(states['simpos'], times)
    #model.comp['eclipse'].set_data(False)
    model.comp['1pdeaat'].set_data(T_psmc, T_psmc_times)
    # 1PIN1AT is broken, so we set its initial condition
    # using an offset, which makes sense based on historical
    # data
    if T_pin1at is None:
        T_pin1at = model.comp["1pdeaat"].dvals - 10.0
    model.comp['pin1at'].set_data(T_pin1at,T_pin1at_times)
    model.comp['roll'].set_data(calc_off_nom_rolls(states), times)
    model.comp['eclipse'].set_data(False)
    for name in ('ccd_count', 'fep_count', 'vid_board', 'clocking', 'pitch'):
        model.comp[name].set_data(states[name], times)
    model.comp['dh_heater'].set_data(dh_heater,dh_heater_times)
    model.make()
    model.calc()
    return model

class PSMCModelCheck(ACISThermalCheck):

    def set_initial_state(self, tlm, db):
        state0 = cmd_states.get_state0(DateTime(tlm['date'][-5]).date, db,
                                           datepar='datestart', date_margin=-100)
        ok = ((tlm['date'] >= state0['tstart'] - 700) &
              (tlm['date'] <= state0['tstart'] + 700))
        state0.update({self.t_msid: np.mean(tlm[self.msid][ok])})
        return state0

    def calc_model_wrapper(self, oflsdir, model_spec, states, tstart, tstop, state0=None):
        if state0 is None:
            start_msid = None
            start_pin = None
            dh_heater = None
            dh_heater_times = None
        else:
            start_msid = state0[self.t_msid]
            start_pin = state0[self.t_msid]-10.0 # the infamous pin1at hack
            htrbfn = os.path.join(oflsdir, 'dahtbon_history.rdb')
            logger.info('Reading file of dahtrb commands from file %s' % htrbfn)
            htrb = ascii.read(htrbfn, format='rdb')
            dh_heater_times = date2secs(htrb['time'])
            dh_heater = htrb['dahtbon'].astype(bool)
        return self.calc_model(model_spec, states, tstart, tstop, T_psmc=start_msid,
                               T_psmc_times=None, T_pin1at=start_pin, T_pin1at_times=None,
                               dh_heater=dh_heater, dh_heater_times=dh_heater_times)

    def write_states(self, opt, states):
        super(PSMCModelCheck, self).write_states(opt, states, remove_cols=['T_pin1at'])

psmc_check = PSMCModelCheck("1pdeaat", "psmc", MSID,
                            YELLOW, MARGIN, VALIDATION_LIMITS,
                            HIST_LIMIT, calc_model,
                            other_telem=['1dahtbon'],
                            other_map={'1dahtbon': 'dh_heater'},
                            other_opts=['dh_heater'])

def main():
    dhh_opt = {"type": int, "default": 0,
               "help": "Starting Detector Housing Heater state"}
    args = get_options("1PDEAAT", "psmc", model_path, [("dh_heater", dhh_opt)])
    try:
        psmc_check.driver(args)
    except Exception as msg:
        if args.traceback:
            raise
        else:
            print("ERROR:", msg)
            sys.exit(1)

if __name__ == '__main__':
    main()
