import numpy as np
from numpy import linalg
import sys
import scipy.sparse as sp
from scipy.stats import norm
from scipy.linalg import solve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
np.set_printoptions(threshold=sys.maxsize)
import skfmm
from scipy.sparse import spdiags, kron, eye


def tau_g(a, b, t):
    return a * np.exp(-b * t)


# compute the evolution of volume following a Gompertz law
def gompertz_area(a, b, t, A0):
    return A0 * np.exp(a*(1 - np.exp(-b*t)) / b)


# build the source term for the spheroid growth
def build_source_term(t, P, Pc, a, b, d):
    f = tau_g(a, b, t) * P - d * Pc
    f[:, 0] = 0
    if f.shape[1] > 1:
        f[:, f.shape[1]-1] = 0
    else:
        f = np.zeros_like(f)
    f[0, :] = 0
    f[f.shape[0]-1, :] = 0
    return f


# build the matrix of discrete laplacian with the K tensor
def build_laplacian(nx, nz, dx, dz, K):
    K_11, K_12, K_21, K_22 = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
    beta = [(K_11 + K_22) / dx ** 2, (K_11 + K_22) / dz ** 2]
    ex = np.ones(nx)
    ez = np.ones(nz)

    Dxx = spdiags([(K_11 / dz ** 2)*ex, -beta[0] * ex, (K_11 / dz ** 2)*ex], [-1, 0, 1], nx, nx)
    Dzz = spdiags([(K_22 / dx ** 2)*ez, -beta[1] * ez, (K_22 / dx ** 2)*ez], [-1, 0, 1], nz, nz)

    L = kron(Dxx, eye(nz)) + kron(eye(nx), Dzz)

    L_array = L.todense()

    # Dirichlet BCs
    for it in np.arange(nx * nz):
        if (it <= nz) or (it >= nx*nz - nz):
            L_array[it, :] = 0.
            L_array[it, it] = 1.
        if (it % nz == 0) and it != 0:
            L_array[it, :] = 0.
            L_array[it, it] = 1.
            L_array[it-1, :] = 0.
            L_array[it-1, it-1] = 1.

    return L_array


def model(physical_param, time_param, space_param):
        
    # time discretization
    time = np.linspace(0, time_param["tmax"], time_param["N_full_time"])
    dt = time_param["tmax"] / (time_param["N_full_time"] - 1)
    t = time[0]
    
    # space discretization
    x = np.linspace(0., space_param["lx"], space_param["Nx"])
    z = np.linspace(0., space_param["lz"], space_param["Nz"])
    dx = x[1] - x[0]
    dz = z[1] - z[0]

    # /!\ meshgrid indexing is different than FE indexing > therefore the final results are transposed
    X, Z = np.meshgrid(x, z, indexing='xy')
    phi = np.sqrt((X - space_param["lx"]/2.) ** 2 + (Z - space_param["lz"]/2.) ** 2) - physical_param["R0"]

    P = physical_param["P0"]*(phi < 0)
    Pc = (1-physical_param["P0"])*(phi < 0)
    p = physical_param["P0"]
    pc = 1-physical_param["P0"]
    
    # build K
    K = np.array([[physical_param["k11"],physical_param["k12"]],[physical_param["k21"],physical_param["k22"]]])    
    
    # initial plots and areas
    list_plots_P = []
    list_plots_Pc = []
    list_areas = []
    list_p = []
    list_pc = []
    
    
    for i in range(time_param["N_full_time"]):

        # build area
        area = np.sum(1.0*(phi < 0))*dx*dz
        
        # print time & area
        print("Time", t, " & Area", area)
        list_areas.append(area)
        list_p.append(p)
        list_pc.append(pc)
        if i in time_param["list_time"]:
            list_plots_P.append(np.transpose(P))
            list_plots_Pc.append(np.transpose(Pc))

        # solve Poisson equation for the pressure:
        # - div (K grad p ) = f + Dirichlet conditions
        A = - build_laplacian(space_param["Nx"], space_param["Nz"], dx, dz, K)

        source = build_source_term(t, P, Pc, physical_param["a"], physical_param["b"], physical_param["d"])
        pressure = solve(A, source.flatten())
        pressure = pressure.reshape((space_param["Nx"], space_param["Nz"]))
        
        # build velocity 
        # v = - K grad p
        dpressurex, dpressurez = np.gradient(pressure, dx, dz, edge_order=2)
        vx = - K[0, 0]*dpressurex - K[0, 1]*dpressurez
        vz = - K[1, 0]*dpressurex - K[1, 1]*dpressurez

        dphix, dphiz = np.gradient(phi, dx, dz, edge_order=2)

        # compute v . grad(phi)
        vgradphi = vx*dphix + vz*dphiz

        
        # advance front
        phi = phi - dt * vgradphi

        # compute value of P and Pc 
        p = p + dt * (- physical_param["c"] * p)
        pc = 1 - p
        P = p*(phi < 0.)
        Pc = pc*(phi < 0.)
        
        # update K 
        K[1,1] = max(1e-10,physical_param["k22"]-physical_param["e"]*pc)

        t += dt
        
    # plot    
    nb_subplots = len(time_param["list_time"])
    rows = int(np.sqrt(nb_subplots))
    cols = int(np.ceil(nb_subplots / rows))
   

    #norm = plt.Normalize(np.min(list_plots_P), np.max(list_plots_P))
    cmap = plt.get_cmap('coolwarm')
    sm = plt.cm.ScalarMappable(cmap=cmap) #, norm=norm)
    sm.set_array([])

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})

    im = axes[0][0].imshow(list_plots_Pc[0], cmap='coolwarm')

    for i, it in enumerate(time_param["list_time"]):
            row = i // cols
            col = i % cols
            ax = axes[row][col]
            im = ax.imshow(list_plots_P[i]+list_plots_Pc[i], cmap='coolwarm', extent=[0, 1., 0, 1.], vmin=0., vmax=1.)
            ax.set_xlabel('x')
            ax.set_ylabel('z')
            ax.set_title('time = {}'.format(round(time[it], 3)))

    if nb_subplots < rows * cols:
        erase = rows * cols - nb_subplots
        for j in range(1, erase + 1):
            fig.delaxes(axes.flatten()[-j])

    fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.4)
    plt.savefig('Results/Complex/tumor.pdf')

    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})

    im = axes[0][0].imshow(list_plots_P[0], cmap='coolwarm')

    for i, it in enumerate(time_param["list_time"]):
        row = i // cols
        col = i % cols
        ax = axes[row][col]
        im = ax.imshow(list_plots_P[i], cmap='coolwarm', extent=[0, 1., 0, 1.], vmin=0., vmax=1.)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('time = {}'.format(round(time[it], 3)))

    if nb_subplots < rows * cols:
        erase = rows * cols - nb_subplots
        for j in range(1, erase + 1):
            fig.delaxes(axes.flatten()[-j])

    # fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.4)
    fig.colorbar(im, ax=axes.ravel().tolist(), location="right", shrink=0.4)
    plt.savefig('Results/Complex/tumorP.pdf')
    
    
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 20), gridspec_kw={'wspace': 0.4, 'hspace': 0.4})

    im = axes[0][0].imshow(list_plots_Pc[0], cmap='coolwarm')

    for i, it in enumerate(time_param["list_time"]):
        # print("Pc = ", list_plots_Pc[i])
        row = i // cols
        col = i % cols
        ax = axes[row][col]
        im = ax.imshow(list_plots_Pc[i], cmap='coolwarm', extent=[0, 1., 0, 1.], vmin=0., vmax=1.)
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title('time = {}'.format(round(time[it], 3)))

    if nb_subplots < rows * cols:
        erase = rows * cols - nb_subplots
        for j in range(1, erase + 1):
            fig.delaxes(axes.flatten()[-j])

    # fig.colorbar(sm, ax=axes.ravel().tolist(), location="right", shrink=0.4)
    fig.colorbar(im, ax=axes.ravel().tolist(), location="right", shrink=0.4)
    plt.savefig('Results/Complex/tumorPc.pdf')
    
    return time, list_areas, list_p, list_pc


# physical quantities
physical_param = {"a": 0.117, "b": 0.14, "c": 0.05, "d": 0.02, "e": 1, "R0": 0.37, "P0": 1, "k11": 1, "k12": 0, "k21": 0, "k22": 0.1}
#time
time_param = {"tmax": 30, "N_full_time": 100, "list_time": np.array([0, 5, 10, 15, 20, 25, 30])}
#mesh
space_param = {"Nx": 61, "Nz": 61, "lx": 2, "lz": 2}

# simple model computation
time, computed_areas, p, pc = model(physical_param, time_param, space_param)

gompertz_areas = gompertz_area(physical_param["a"], physical_param["b"], time, computed_areas[0])

fig = plt.figure()
plt.plot(time, computed_areas, label='Computed areas')
plt.plot(time, gompertz_areas, label='Gompertz areas')
plt.xlabel('Time')
plt.ylabel('Area')
plt.legend()
plt.savefig('Results/Complex/areas.pdf')


fig = plt.figure()
plt.plot(time, p, label='P areas')
plt.plot(time, pc, label='P_c')
plt.xlabel('time')
plt.ylabel('cell')
plt.legend()
plt.savefig('Results/Complex/cell.pdf')
