# importing useful packages
import numpy as np
from scipy.integrate import trapz
from scipy.stats import norm


def model(tmax, R0, P0mean, b, k, p_of_N):
    # space discretization
    N_mesh = 200
    r01 = np.linspace(0, 1, N_mesh)
    dr = r01[1] - r01[0]

    # time discretization
    N_full_time = 135
    t = 0.
    full_time = np.linspace(t, tmax, N_full_time)
    dt = tmax / (N_full_time - 1)

    # Parameters (correlation between the parameters a and b in Gompertz model)
    a = k * b

    # Description of N
    Nbegin = p_of_N[0]
    Nend = p_of_N[1]
    dist = p_of_N[2]
    tau = p_of_N[3]

    # Initial condition
    P0 = norm.rvs(loc=P0mean, scale=0.00, size=N_mesh)
    Pn = P0

    # Build radius for each timestep
    R = np.zeros(full_time.shape)

    # Create F, P and N
    P = np.zeros((N_mesh, N_full_time))
    tauP2Q = np.zeros((N_mesh, N_full_time))
    # Initialize
    tauP2Qn = Nbegin - (Nbegin - Nend) / (1 + np.exp((R0 * (1 - r01) - dist) / tau))
    P[:, 0] = Pn
    R[0] = R0

    # Loop
    for nt in range(1, N_full_time):
        # compute R(tn)
        I1 = trapz(r01 ** 2, r01)
        R[nt] = R[nt - 1] * (1 + dt * (a * np.exp(-b * full_time[nt - 1])) * I1)

        # compute M(tn)
        tauGn = a * np.exp(-b * full_time[nt])
        tauP2Qn = Nbegin - (Nbegin - Nend) / (1 + np.exp((R[nt] * (1 - r01) - dist) / tau))

        # compute P(tn, r)
        Pnnew = np.zeros(N_mesh)
        for i in range(N_mesh):
            Ir = np.trapz(np.multiply((r01 <= r01[i]), np.power(r01, 2)), r01)
            if i == 0:
                vgradP = 0.
            else:
                tildevonR = tauGn * ((1 / (r01[i] ** 2)) * Ir - r01[i] * I1)
                vgradP = tildevonR * (Pn[i] - Pn[i - 1]) / dr
            Pnnew[i] = Pn[i] + dt * (-vgradP + tauGn * (1 - Pn[i]) - tauP2Qn[i] * Pn[i])


        Pn = Pnnew
        P[:, nt] = Pn
        tauP2Q[:, nt] = tauP2Qn

    return full_time, R, r01, P
