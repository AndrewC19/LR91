# Size of variable arrays:
SIZE_ALGEBRAIC = 25
SIZE_STATES = 8
SIZE_CONSTANTS = 24

# Configuration of ODE solver
START_TIME = 8000
END_TIME = 8800
TIME_STEPS = 10000

OUTPUT_PATH = "./data/results.csv"
import matplotlib.pyplot as plt
import pandas as pd
from math import *
from numpy import *


def custom_piecewise(cases):
    """Compute result of a piecewise function"""
    return select(cases[0::2], cases[1::2])


def save_results(model_results):
    model_results_df = pd.DataFrame.from_dict(model_results, orient='index')
    model_results_df.to_csv(OUTPUT_PATH)


class LR91:

    def __init__(self):
        self.G_Na = 23.0000
        self.G_si = 0.09000
        self.G_K = 0.28200
        self.G_K1 = 0.60470
        self.G_Kp = 0.01830
        self.G_b = 0.03921

    def sample_conductances(self):
        """ Sample the conductance input parameter values uniformly from the design data distributions specified in
            (Chang, Strong, and Clayton, 2015), Table 1 [1].

            [1] Chang, E.T., Strong, M. and Clayton, R.H., 2015.
                Bayesian sensitivity analysis of a cardiac cell model using a Gaussian process emulator.
                PloS one, 10(6), p.e0130252.
                https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130252
        """
        self.G_Na = random.uniform(17.250, 28.750)
        self.G_si = random.uniform(0.0675, 0.1125)
        self.G_K = random.uniform(0.2115, 0.3525)
        self.G_K1 = random.uniform(0.4535, 0.7559)
        self.G_Kp = random.uniform(0.0137, 0.0229)
        self.G_b = random.uniform(0.0294, 0.0490)

    def create_legends(self):
        legend_states = [""] * SIZE_STATES
        legend_rates = [""] * SIZE_STATES
        legend_algebraic = [""] * SIZE_ALGEBRAIC
        legend_voi = ""
        legend_constants = [""] * SIZE_CONSTANTS
        legend_voi = "Time in component environment $(ms)$"
        legend_states[0] = "$V$ in component membrane $(mV)$"
        legend_constants[0] = "$R$ in component membrane $(J/kM/K)$"
        legend_constants[1] = "$T$ in component membrane $(K)$"
        legend_constants[2] = "$F$ in component membrane $(C/M)$"
        legend_constants[3] = "$C$ in component membrane $(\mu F/cm^2)$"
        legend_algebraic[0] = "$I_{stim}$ in component membrane $(\mu A/cm^2)$"
        legend_algebraic[7] = "$i_{Na}$ in component fast sodium current $(\mu A/cm^2)$"
        legend_algebraic[15] = "$i_{si}$ in component slow inward current $(\mu A/cm^2)$"
        legend_algebraic[17] = "$i_K$ in component time dependent potassium current $(\mu A/cm^2)$"
        legend_algebraic[21] = "$i_{K1}$ in component time independent potassium current $(\mu A/cm^2)$"
        legend_algebraic[23] = "$i_{Kp}$ in component plateau potassium current $(\mu A/cm^2)$"
        legend_algebraic[24] = "$i_b$ in component background current $(\mu A/cm^2)$"
        legend_constants[4] = "stim start in component membrane $(ms)$"
        legend_constants[5] = "stim end in component membrane $(ms)$"
        legend_constants[6] = "stim period in component membrane $(ms)$"
        legend_constants[7] = "stim duration in component membrane $(ms)$"
        legend_constants[8] = "stim amplitude in component membrane $(\mu A/cm^2)$"
        legend_constants[9] = "$g_{Na}$ in component fast sodium current $(mS/cm^2)$"
        legend_constants[18] = "$E_{Na}$ in component fast sodium current $(mV)$"
        legend_constants[10] = "$Nao$ in component ionic concentrations $(mM)$"
        legend_constants[11] = "$Nai$ in component ionic concentrations $(mM)$"
        legend_states[1] = "$m$ in component fast sodium current $m$ gate (dimensionless)"
        legend_states[2] = "$h$ in component fast sodium current $h$ gate (dimensionless)"
        legend_states[3] = "$j$ in component fast sodium current $j$ gate (dimensionless)"
        legend_algebraic[1] = r"$\alpha_m$ in component fast sodium current $m$ gate $(ms^{-1})$"
        legend_algebraic[8] = r"$\beta_m$ in component fast sodium current $m$ gate $(ms^{-1})$"
        legend_algebraic[2] = r"$\alpha_h$ in component fast sodium current $h$ gate $(ms^{-1})$"
        legend_algebraic[9] = r"$\beta_h$ in component fast sodium current $h$ gate $(ms^{-1})$"
        legend_algebraic[3] = r"$\alpha_j$ in component fast sodium current $j$ gate $(ms^{-1})$"
        legend_algebraic[10] = r"$\beta_j$ in component fast sodium current $j$ gate $(ms^{-1})$"
        legend_algebraic[14] = "$E_{si}$ in component slow inward current $(mV)$"
        legend_states[4] = "$Cai$ in component intracellular calcium concentration $(mM)$"
        legend_states[5] = "$d$ in component slow inward current $d$ gate (dimensionless)"
        legend_states[6] = "$f$ in component slow inward current $f$ gate (dimensionless)"
        legend_algebraic[4] = r"$\alpha_d$ in component slow inward current $d$ gate $(ms^{-1})$"
        legend_algebraic[11] = r"$\beta_d$ in component slow inward current $d$ gate $(ms^{-1})$"
        legend_algebraic[5] = r"$\alpha_f$ in component slow inward current $f$ gate $(ms^{-1})$"
        legend_algebraic[12] = r"$\beta_f$ in component slow inward current $f$ gate $(ms^{-1})$"
        legend_constants[19] = "$g_K$ in component time dependent potassium current $(mS/cm^2)$"
        legend_constants[20] = "$E_K$ in component time dependent potassium current $(mV)$"
        legend_constants[12] = "$PR_{NaK}$ in component time dependent potassium current (dimensionless)"
        legend_constants[13] = "$Ko$ in component ionic concentrations $(mM)$"
        legend_constants[14] = "$Ki$ in component ionic concentrations $(mM)$"
        legend_states[7] = "$X$ in component time dependent potassium current $X$ gate (dimensionless)"
        legend_algebraic[16] = "$Xi$ in component time dependent potassium current $Xi$ gate (dimensionless)"
        legend_algebraic[6] = r"$\alpha_X$ in component time dependent potassium current $X$ gate $(ms^{-1})$"
        legend_algebraic[13] = r"$\beta_X$ in component time dependent potassium current $X$ gate $(ms^{-1})$"
        legend_constants[21] = "$E_{K1}$ in component time independent potassium current $(mV)$"
        legend_constants[22] = "$g_{K1}$ in component time independent potassium current $(mS/cm^2)$"
        legend_algebraic[
            20] = "$K1_{infinity}$ in component time independent potassium current $K1$ gate (dimensionless)"
        legend_algebraic[18] = r"$\alpha_K1$ in component time independent potassium current $K1$ gate $(ms^{-1})$"
        legend_algebraic[19] = r"$\beta_K1$ in component time independent potassium current $K1$ gate $(ms^{-1})$"
        legend_constants[23] = "$E_Kp$ in component plateau potassium current $(mV)$"
        legend_constants[15] = "$g_Kp$ in component plateau potassium current $(mS/cm^2)$"
        legend_algebraic[22] = "$Kp$ in component plateau potassium current (dimensionless)"
        legend_constants[16] = "$E_b$ in component background current $(mV)$"
        legend_constants[17] = "$g_b$ in component background current $(mS/cm^2)$"
        legend_rates[0] = r"$\frac{d}{dt}$ V in component membrane $(mV)$"
        legend_rates[1] = r"$\frac{d}{dt}$ $m$ in component fast sodium current $m$ gate (dimensionless)"
        legend_rates[2] = r"$\frac{d}{dt}$ $h$ in component fast sodium current $h$ gate (dimensionless)"
        legend_rates[3] = r"$\frac{d}{dt}$ $j$ in component fast sodium current $j$ gate (dimensionless)"
        legend_rates[5] = r"$\frac{d}{dt}$ $d$ in component slow inward current $d$ gate (dimensionless)"
        legend_rates[6] = r"$\frac{d}{dt}$ $f$ in component slow inward current $f$ gate (dimensionless)"
        legend_rates[7] = r"$\frac{d}{dt}$ $X$ in component time dependent potassium current $X$ gate (dimensionless)"
        legend_rates[4] = r"$\frac{d}{dt}$ $Cai$ in component intracellular calcium concentration $(mM)$"
        return (legend_states, legend_algebraic, legend_voi, legend_constants)

    def initialise_constants(self):
        constants = [0.0] * SIZE_CONSTANTS
        states = [0.0] * SIZE_STATES
        states[0] = -84.3801107371
        constants[0] = 8314
        constants[1] = 310
        constants[2] = 96484.6
        constants[3] = 1
        constants[4] = 100  # Stim start
        constants[5] = 9000  # Stim end
        constants[6] = 1000  # Stim period
        constants[7] = 2  # Stim duration
        constants[8] = -25.5  # Stim strength
        constants[9] = self.G_Na
        constants[10] = 140
        constants[11] = 18
        states[1] = 0.00171338077730188
        states[2] = 0.982660523699656
        states[3] = 0.989108212766685
        states[4] = 0.00017948816388306
        states[5] = 0.00302126301779861
        states[6] = 0.999967936476325
        constants[12] = 0.01833
        constants[13] = 5.4
        constants[14] = 145
        states[7] = 0.0417603108167287
        constants[15] = self.G_Kp
        constants[16] = -59.87

        constants[17] = self.G_b
        constants[18] = ((constants[0] * constants[1]) / constants[2]) * log(constants[10] / constants[11])
        constants[19] = self.G_K * (power(constants[13] / 5.40000, 1.0 / 2))
        constants[20] = ((constants[0] * constants[1]) / constants[2]) * log(
            (constants[13] + constants[12] * constants[10]) / (constants[14] + constants[12] * constants[11]))
        constants[21] = ((constants[0] * constants[1]) / constants[2]) * log(constants[13] / constants[14])
        constants[22] = self.G_K1 * (power(constants[13] / 5.40000, 1.0 / 2))
        constants[23] = constants[21]
        return (states, constants)

    def compute_rates(self, voi, states, constants):
        rates = [0.0] * SIZE_STATES
        algebraic = [0.0] * SIZE_ALGEBRAIC
        algebraic[1] = (0.320000 * (states[0] + 47.1300)) / (1.00000 - exp(-0.100000 * (states[0] + 47.1300)))
        algebraic[8] = 0.0800000 * exp(-states[0] / 11.0000)
        rates[1] = algebraic[1] * (1.00000 - states[1]) - algebraic[8] * states[1]
        algebraic[2] = custom_piecewise(
            [less(states[0], -40.0000), 0.135000 * exp((80.0000 + states[0]) / -6.80000), True, 0.00000])
        algebraic[9] = custom_piecewise(
            [less(states[0], -40.0000), 3.56000 * exp(0.0790000 * states[0]) + 310000. * exp(0.350000 * states[0]),
             True, 1.00000 / (0.130000 * (1.00000 + exp((states[0] + 10.6600) / -11.1000)))])
        rates[2] = algebraic[2] * (1.00000 - states[2]) - algebraic[9] * states[2]
        algebraic[3] = custom_piecewise([less(states[0], -40.0000), (
                    (-127140. * exp(0.244400 * states[0]) - 3.47400e-05 * exp(-0.0439100 * states[0])) * (
                        states[0] + 37.7800)) / (1.00000 + exp(0.311000 * (states[0] + 79.2300))), True, 0.00000])
        algebraic[10] = custom_piecewise([less(states[0], -40.0000), (0.121200 * exp(-0.0105200 * states[0])) / (
                    1.00000 + exp(-0.137800 * (states[0] + 40.1400))), True,
                                          (0.300000 * exp(-2.53500e-07 * states[0])) / (
                                                      1.00000 + exp(-0.100000 * (states[0] + 32.0000)))])
        rates[3] = algebraic[3] * (1.00000 - states[3]) - algebraic[10] * states[3]
        algebraic[4] = (0.0950000 * exp(-0.0100000 * (states[0] - 5.00000))) / (
                    1.00000 + exp(-0.0720000 * (states[0] - 5.00000)))
        algebraic[11] = (0.0700000 * exp(-0.0170000 * (states[0] + 44.0000))) / (
                    1.00000 + exp(0.0500000 * (states[0] + 44.0000)))
        rates[5] = algebraic[4] * (1.00000 - states[5]) - algebraic[11] * states[5]
        algebraic[5] = (0.0120000 * exp(-0.00800000 * (states[0] + 28.0000))) / (
                    1.00000 + exp(0.150000 * (states[0] + 28.0000)))
        algebraic[12] = (0.00650000 * exp(-0.0200000 * (states[0] + 30.0000))) / (
                    1.00000 + exp(-0.200000 * (states[0] + 30.0000)))
        rates[6] = algebraic[5] * (1.00000 - states[6]) - algebraic[12] * states[6]
        algebraic[6] = (0.000500000 * exp(0.0830000 * (states[0] + 50.0000))) / (
                    1.00000 + exp(0.0570000 * (states[0] + 50.0000)))
        algebraic[13] = (0.00130000 * exp(-0.0600000 * (states[0] + 20.0000))) / (
                    1.00000 + exp(-0.0400000 * (states[0] + 20.0000)))
        rates[7] = algebraic[6] * (1.00000 - states[7]) - algebraic[13] * states[7]
        algebraic[14] = 7.70000 - 13.0287 * log(states[4] / 1.00000)
        algebraic[15] = 0.0900000 * states[5] * states[6] * (states[0] - algebraic[14])
        rates[4] = (-0.000100000 / 1.00000) * algebraic[15] + 0.0700000 * (0.000100000 - states[4])
        algebraic[0] = custom_piecewise([greater_equal(voi, constants[4]) & less_equal(voi, constants[5]) & less_equal(
            (voi - constants[4]) - floor((voi - constants[4]) / constants[6]) * constants[6], constants[7]),
                                         constants[8], True, 0.00000])
        algebraic[7] = constants[9] * (power(states[1], 3.00000)) * states[2] * states[3] * (states[0] - constants[18])
        algebraic[16] = custom_piecewise([greater(states[0], -100.000),
                                          (2.83700 * (exp(0.0400000 * (states[0] + 77.0000)) - 1.00000)) / (
                                                      (states[0] + 77.0000) * exp(0.0400000 * (states[0] + 35.0000))),
                                          True, 1.00000])
        algebraic[17] = constants[19] * states[7] * algebraic[16] * (states[0] - constants[20])
        algebraic[18] = 1.02000 / (1.00000 + exp(0.238500 * ((states[0] - constants[21]) - 59.2150)))
        algebraic[19] = (0.491240 * exp(0.0803200 * ((states[0] + 5.47600) - constants[21])) + 1.00000 * exp(
            0.0617500 * (states[0] - (constants[21] + 594.310)))) / (
                                    1.00000 + exp(-0.514300 * ((states[0] - constants[21]) + 4.75300)))
        algebraic[20] = algebraic[18] / (algebraic[18] + algebraic[19])
        algebraic[21] = constants[22] * algebraic[20] * (states[0] - constants[21])
        algebraic[22] = 1.00000 / (1.00000 + exp((7.48800 - states[0]) / 5.98000))
        algebraic[23] = constants[15] * algebraic[22] * (states[0] - constants[23])
        algebraic[24] = constants[17] * (states[0] - constants[16])
        rates[0] = (-1.00000 / constants[3]) * (
                    algebraic[0] + algebraic[7] + algebraic[15] + algebraic[17] + algebraic[21] + algebraic[23] +
                    algebraic[24])
        return (rates)

    def compute_algebraic(self, constants, states, voi):
        algebraic = array([[0.0] * len(voi)] * SIZE_ALGEBRAIC)
        states = array(states)
        voi = array(voi)
        algebraic[1] = (0.320000 * (states[0] + 47.1300)) / (1.00000 - exp(-0.100000 * (states[0] + 47.1300)))
        algebraic[8] = 0.0800000 * exp(-states[0] / 11.0000)
        algebraic[2] = custom_piecewise(
            [less(states[0], -40.0000), 0.135000 * exp((80.0000 + states[0]) / -6.80000), True, 0.00000])
        algebraic[9] = custom_piecewise(
            [less(states[0], -40.0000), 3.56000 * exp(0.0790000 * states[0]) + 310000. * exp(0.350000 * states[0]),
             True, 1.00000 / (0.130000 * (1.00000 + exp((states[0] + 10.6600) / -11.1000)))])
        algebraic[3] = custom_piecewise([less(states[0], -40.0000), (
                    (-127140. * exp(0.244400 * states[0]) - 3.47400e-05 * exp(-0.0439100 * states[0])) * (
                        states[0] + 37.7800)) / (1.00000 + exp(0.311000 * (states[0] + 79.2300))), True, 0.00000])
        algebraic[10] = custom_piecewise([less(states[0], -40.0000), (0.121200 * exp(-0.0105200 * states[0])) / (
                    1.00000 + exp(-0.137800 * (states[0] + 40.1400))), True,
                                          (0.300000 * exp(-2.53500e-07 * states[0])) / (
                                                      1.00000 + exp(-0.100000 * (states[0] + 32.0000)))])
        algebraic[4] = (0.0950000 * exp(-0.0100000 * (states[0] - 5.00000))) / (
                    1.00000 + exp(-0.0720000 * (states[0] - 5.00000)))
        algebraic[11] = (0.0700000 * exp(-0.0170000 * (states[0] + 44.0000))) / (
                    1.00000 + exp(0.0500000 * (states[0] + 44.0000)))
        algebraic[5] = (0.0120000 * exp(-0.00800000 * (states[0] + 28.0000))) / (
                    1.00000 + exp(0.150000 * (states[0] + 28.0000)))
        algebraic[12] = (0.00650000 * exp(-0.0200000 * (states[0] + 30.0000))) / (
                    1.00000 + exp(-0.200000 * (states[0] + 30.0000)))
        algebraic[6] = (0.000500000 * exp(0.0830000 * (states[0] + 50.0000))) / (
                    1.00000 + exp(0.0570000 * (states[0] + 50.0000)))
        algebraic[13] = (0.00130000 * exp(-0.0600000 * (states[0] + 20.0000))) / (
                    1.00000 + exp(-0.0400000 * (states[0] + 20.0000)))
        algebraic[14] = 7.70000 - 13.0287 * log(states[4] / 1.00000)
        algebraic[15] = 0.0900000 * states[5] * states[6] * (states[0] - algebraic[14])
        algebraic[0] = custom_piecewise([greater_equal(voi, constants[4]) & less_equal(voi, constants[5]) & less_equal(
            (voi - constants[4]) - floor((voi - constants[4]) / constants[6]) * constants[6], constants[7]),
                                         constants[8], True, 0.00000])
        algebraic[7] = constants[9] * (power(states[1], 3.00000)) * states[2] * states[3] * (states[0] - constants[18])
        algebraic[16] = custom_piecewise([greater(states[0], -100.000),
                                          (2.83700 * (exp(0.0400000 * (states[0] + 77.0000)) - 1.00000)) / (
                                                      (states[0] + 77.0000) * exp(0.0400000 * (states[0] + 35.0000))),
                                          True, 1.00000])
        algebraic[17] = constants[19] * states[7] * algebraic[16] * (states[0] - constants[20])
        algebraic[18] = 1.02000 / (1.00000 + exp(0.238500 * ((states[0] - constants[21]) - 59.2150)))
        algebraic[19] = (0.491240 * exp(0.0803200 * ((states[0] + 5.47600) - constants[21])) + 1.00000 * exp(
            0.0617500 * (states[0] - (constants[21] + 594.310)))) / (
                                    1.00000 + exp(-0.514300 * ((states[0] - constants[21]) + 4.75300)))
        algebraic[20] = algebraic[18] / (algebraic[18] + algebraic[19])
        algebraic[21] = constants[22] * algebraic[20] * (states[0] - constants[21])
        algebraic[22] = 1.00000 / (1.00000 + exp((7.48800 - states[0]) / 5.98000))
        algebraic[23] = constants[15] * algebraic[22] * (states[0] - constants[23])
        algebraic[24] = constants[17] * (states[0] - constants[16])
        return algebraic

    def run_model(self, plot=True, sample_conductances=False, n_runs=1, verbose=False, detailed_plot=False):
        """
        Run the LR91 model.
        :param verbose: Whether or not to print verbose details (model run progress).
        :param n_runs: Number of model runs.
        :param plot: Whether or not to produce a plot (boolean)
        :param sample_conductances: Whether or not to sample conductances (boolean)
        :return: a dictionary of model run data, including inputs, voi, states, and algebraic values
        """
        model_runs = {}
        for n_run in range(n_runs):
            if verbose:
                print(f"Model run {n_run + 1}/{n_runs}")
            if sample_conductances:
                self.sample_conductances()
            voi, states, algebraic = self.solve_model()
            model_runs[n_run] = {"G_Na": self.G_Na, "G_si": self.G_si, "G_K": self.G_K, "G_K1": self.G_K1,
                                 "G_Kp": self.G_Kp, "G_b": self.G_b, "voi": voi, "states": states,
                                 "algebraic": algebraic}
        model_results = self.compute_voltage_outputs(model_runs)
        save_results(model_results)
        if plot:
            # self.plot_model(voi, states, algebraic)
            self.plot_voltage_from_model_runs(model_runs, model_results, detailed_plot)
        return model_runs

    def compute_voltage_outputs(self, model_runs):
        model_results = {}
        for model_run, run_results in model_runs.items():
            model_results[model_run] = {key: run_results[key] for key in run_results if key not in ["voi", "states",
                                                                                                    "algebraic"]}
            model_results[model_run]['max_voltage'] = self.compute_max_voltage(run_results)
            model_results[model_run]['rest_voltage'] = self.compute_rest_voltage(run_results)
            model_results[model_run]['max_voltage_gradient'] = self.compute_max_voltage_gradient(run_results)
            model_results[model_run]['dome_voltage'] = self.compute_dome_voltage(run_results)
            model_results[model_run]['APD50'] = self.compute_APDx(run_results, model_results[model_run]['rest_voltage'],
                                                                  model_results[model_run]['max_voltage'], 50)
            model_results[model_run]['APD90'] = self.compute_APDx(run_results, model_results[model_run]['rest_voltage'],
                                                                  model_results[model_run]['max_voltage'], 90)
        return model_results

    def solve_model(self):
        """ Solve model with ODE solver. """
        from scipy.integrate import ode
        # Initialise constants and state variables
        init_states, constants = self.initialise_constants()

        # Set timespan to solve over
        voi = linspace(START_TIME, END_TIME, TIME_STEPS)

        # Construct ODE object to solve
        r = ode(self.compute_rates)
        r.set_integrator('vode', method='bdf', atol=1e-06, rtol=1e-06, max_step=1)
        r.set_initial_value(init_states, voi[0])
        r.set_f_params(constants)

        # Solve model
        states = array([[0.0] * len(voi)] * SIZE_STATES)
        states[:, 0] = init_states
        for (i, t) in enumerate(voi[1:]):
            if r.successful():
                r.integrate(t)
                states[:, i + 1] = r.y
            else:
                break

        # Compute algebraic variables
        algebraic = self.compute_algebraic(constants, states, voi)
        return voi, states, algebraic

    def plot_model(self, voi, states, algebraic):
        """
        Plot variables against variable of integration.
        :param voi: Variables of integration.
        :param states: State variables.
        :param algebraic: Algebraic variables.
        """

        legend_states, legend_algebraic, legend_voi, legend_constants = self.create_legends()
        plt.figure(1, figsize=(14.8, 11.8))
        ax = plt.subplot(111)
        plt.plot(voi, states.T, linestyle='-')
        plt.plot(voi, algebraic.T, linestyle='--')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height * 0.8])
        plt.xlabel(legend_voi)
        algebraic_legend = plt.legend(legend_algebraic, title="Algebraics", loc='lower right',
                                      bbox_to_anchor=(2.14, -0.01))
        states_legend = plt.legend(legend_states, title="Constants", loc='lower left', bbox_to_anchor=(-0.01, 1))
        plt.gca().add_artist(algebraic_legend)
        plt.gca().add_artist(states_legend)
        plt.show()

    @staticmethod
    def plot_voltage_from_model_runs(model_runs, model_results, plot_model_results=False):
        """
        Plot voltage against variable of integration.
        :param plot_model_results: A bool that determines whether model results are plotted.
        :param model_results: Model results to plot.
        :param model_runs: A dictionary of individual model runs.
        """

        plt.figure()
        for model_run, run_results in model_runs.items():
            xs = run_results["voi"]
            ys = run_results["states"][0].T
            plt.plot(xs, ys, color='b', alpha=.8)
            if plot_model_results:
                dome_voltage = model_results[model_run]["dome_voltage"]
                max_voltage = model_results[model_run]["max_voltage"]
                rest_voltage = model_results[model_run]["rest_voltage"]
                action_potential = max_voltage - rest_voltage
                apd_50 = int(model_results[model_run]["APD50"] * (TIME_STEPS / (END_TIME - START_TIME)))
                apd_90 = int(model_results[model_run]["APD90"] * (TIME_STEPS / (END_TIME - START_TIME)))
                max_voltage_time = list(run_results["states"][0].T).index(max_voltage)
                apd_50s = xs[max_voltage_time:max_voltage_time + apd_50]
                apd_90s = xs[max_voltage_time:max_voltage_time + apd_90]
                plt.plot(xs, [dome_voltage] * len(ys), color='r', linestyle='--', alpha=.3, label="Dome Voltage")
                plt.plot(xs, [max_voltage] * len(ys), color='g', linestyle='--', alpha=.3, label="Max Voltage")
                plt.plot(xs, [rest_voltage] * len(ys), color='black', linestyle='--', alpha=.3, label="Rest Voltage")

                plt.plot(apd_50s, [max_voltage - (action_potential * .5)] * apd_50, color='purple', linestyle='--',
                         alpha=1, label="APD50", markevery=[0, -1], marker="|")
                plt.plot(apd_90s, [max_voltage - (action_potential * .9)] * apd_90, color='orange', linestyle='--',
                         alpha=1, label="APD90", markevery=[0, -1], marker="|")
        plt.xlabel(r"Time $ms$")
        plt.ylabel(r"Voltage $(mV)$")
        plt.legend()
        plt.savefig("lr91.pdf", format="pdf")
        plt.show()
        plt.tight_layout()

    @staticmethod
    def compute_max_voltage(model_run):
        return max(model_run["states"][0].T)

    @staticmethod
    def compute_dome_voltage(model_run):
        voltage_derivatives = gradient(model_run["states"][0].T)
        candidate_turning_points = []
        for i in range(2, len(voltage_derivatives) - 2):
            # Find dome turning point: \_/ left neighbour is negative, right neighbour is positive, turning point must
            # be roughly 0
            # TODO: Fix Dome Voltage to find /-\
            if voltage_derivatives[i - 2] > 0 and voltage_derivatives[i + 2] < 0 and abs(voltage_derivatives[i]) < 0.1:
                candidate_turning_points.append(i)
        candidate_turning_points = [i for i in candidate_turning_points if model_run["states"][0].T[i] > 0]
        turning_point = list(voltage_derivatives).index(min(voltage_derivatives[candidate_turning_points]))
        dome_voltage = model_run["states"][0].T[turning_point]
        return dome_voltage

    @staticmethod
    def compute_APDx(model_run, rest_voltage, max_voltage, percentage):
        """
        Compute the action potential duration to x%. That is, the amount of time until the max voltage falls by
        x%.
        :param model_run: A dictionary of model runs, each containing states, algebraics, and voi.
        :param max_voltage: The max voltage (spike point).
        :param percentage: The percentage of action potential to measure duration to fall to.
        :return: Duration of peak AP to x% APD.
        """
        voltages = model_run["states"][0].T
        action_potential = max_voltage - rest_voltage
        action_potential_x = action_potential * (percentage / 100)
        voltage_x = max_voltage - action_potential_x
        closest_to_voltage_x = min(voltages, key=lambda v: abs(v - voltage_x))
        duration_time_steps = abs(list(voltages).index(closest_to_voltage_x) - list(voltages).index(max_voltage))
        duration_ms = int(duration_time_steps * ((END_TIME - START_TIME) / TIME_STEPS))
        print(duration_time_steps, duration_ms)
        return duration_ms

    @staticmethod
    def compute_max_voltage_gradient(model_run):
        voltage_derivatives = abs(gradient(model_run["states"][0].T))
        # Get the max derivative and scale to be mV per ms
        # To get mV per ms, divide number of time steps by duration
        return max(voltage_derivatives) * (TIME_STEPS / (END_TIME - START_TIME))

    @staticmethod
    def compute_rest_voltage(model_run):
        return model_run["states"][0].T[-1]


if __name__ == "__main__":
    model = LR91()
    results = model.run_model(plot=True, sample_conductances=True, n_runs=1, verbose=True,
                              detailed_plot=True)
