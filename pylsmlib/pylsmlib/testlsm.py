import numpy as np
import pythonisedfns
from pylsmlib import computeDistanceFunction
import matplotlib.pyplot as plt

__docformat__ = 'restructuredtext'

def testCircleEvolution():
    r"""
        Evolves an initially circular level set to see if it remains circular.
    """

    # create circle
    N     = 50
    t     = 0.
    X, Y  = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    r     = 0.1
    dx    = 2.0 / (N - 1)
    dy    = dx
    dt    = 0.001
    phi   = (X) ** 2 + (Y) ** 2 - r ** 2
    phi   = computeDistanceFunction(phi, dx)
    title = "Circle evolution"

    #set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12,9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
                [0, dx*(N-4), 0, dx*(N-4)], [2,N-3, 2, N-3], 0,0)
    plt.show(block=False)

    nIt = 50
    sL0 = 0.01
    marksteinLength = 0.01
    u = np.zeros_like(phi)
    iblims = np.array([2,N-3, 2, N-3])
    lims = np.array([2,N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        #calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi,jlo:jhi] += dt * ( \
            (Fx[ilo:ihi,jlo:jhi] - Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] - Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, lims)

        # plotting
        dovis(phi, title, gridlims, iblims, n, t)
        plt.show(block=False)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

    return


def testSquareEvolution():
    r"""
        Evolves an initially square level set to see if it remains square.
    """

    # create square
    N     = 50
    t     = 0.
    dx    = 2.0 / (N - 1)
    dy    = dx
    dt    = 0.001
    phi   = np.ones((N,N))
    phi[N/2-5:N/2+6,N/2-5:N/2+6] = -1.
    phi   = computeDistanceFunction(phi, dx)
    title = "Square evolution"

    #set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12,9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
                [0, dx*(N-4), 0, dx*(N-4)], [2,N-3, 2, N-3], 0,0)
    plt.show(block=False)

    nIt = 50
    sL0 = 0.01
    marksteinLength = 0.01
    u = np.zeros_like(phi)
    iblims = np.array([2,N-3, 2, N-3])
    lims = np.array([2,N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        #calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi,jlo:jhi] += dt * ( \
            (Fx[ilo:ihi,jlo:jhi] - Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] - Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, lims)

        # plotting
        dovis(phi, title, gridlims, iblims, n, t)
        plt.show(block=False)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

    return


def testSineEvolution():
    r"""
        Evolves level set with periodic boundary conditions.
    """

    # create circle
    N     = 50
    t     = 0.
    r     = 0.1
    dx    = 2.0 / (N - 1)
    dy    = dx
    dt    = 0.1
    phi   = np.ones((N,N))
    phi[N/2-5:N/2+6,:] = -1.
    phi   = computeDistanceFunction(phi, dx)
    title = "Periodic evolution"

    #set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12,9), dpi=100, facecolor='w')
    dovis(phi, title, [0, dx*(N-4), 0, dx*(N-4)], [2,N-3, 2, N-3], 0,0)
    plt.show(block=False)

    nIt = 200
    sL0 = 0.01
    marksteinLength = 0.
    u = np.ones_like(phi)*0.1
    iblims = np.array([2,N-3, 2, N-3])
    lims = np.array([2,N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        #calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u)
        Fx = enforcePeriodicBCs(Fx, lims)
        Fy = enforcePeriodicBCs(Fy, lims)

        #print(Fx[20:-20,30]-Fx[21:-19,30])

        phi[ilo:ihi,jlo:jhi] -= dt * ( \
            (Fx[ilo:ihi,jlo:jhi] + Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] + Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce periodic boundary conditions
        phi[:,:] = enforcePeriodicBCs(phi[:,:], lims)

        # plotting
        dovis(phi, title, gridlims, iblims, n, t)
        plt.show(block=False)

        # reinitialise
        phi[:,:] = computeDistanceFunction(phi[:,:], dx)

    return


def findFluxes(phi, sL, lims, dx=1., dy=1., u=None, v=None):
    r"""
        Find the fluxes.
    """

    ilo, ihi, jlo, jhi = lims
    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)

    if u is None:
        u = np.zeros_like(phi)
    if v is None:
        v = np.zeros_like(phi)

    phi_x, phi_y = pythonisedfns.gradPhi2d(phi, dx=dx, dy=dy)
    norm_x, norm_y = pythonisedfns.signedUnitNormal2d(phi, phi_x, phi_y, dx=dx, dy=dy)

    # normalise norms some more
    norm_x[:,:] /= np.sqrt(norm_x[:,:]**2 + norm_y[:,:]**2)
    norm_y[:,:] /= np.sqrt(norm_x[:,:]**2 + norm_y[:,:]**2)

    # add on laminar flame speed contribution
    u[:,:] += norm_x[:,:] * sL[:,:]
    v[:,:] += norm_y[:,:] * sL[:,:]
    #print(norm_y[:] * sL[:])

    Fx[ilo:ihi, jlo:jhi] = u[ilo:ihi,jlo:jhi] * phi[ilo:ihi,jlo:jhi] - \
                           u[ilo-1:ihi-1,jlo:jhi] * phi[ilo-1:ihi-1,jlo:jhi]
    Fy[ilo:ihi, jlo:jhi] = v[ilo:ihi,jlo:jhi] * phi[ilo:ihi,jlo:jhi] - \
                           v[ilo:ihi,jlo-1:jhi-1] * phi[ilo:ihi,jlo-1:jhi-1]

    return Fx, Fy


def enforceOutflowBCs(phi, lims):
    r"""
        Just copy across outermost cells of inner box
    """
    ilo, ihi, jlo, jhi = lims
    for i in range(2):
        phi[i,jlo:jhi] = phi[2,jlo:jhi]
        phi[-(i+1),jlo:jhi] = phi[-3,jlo:jhi]
        phi[:,i] = phi[:,2]
        phi[:,-(i+1)] = phi[:,-3]
    return phi


def enforcePeriodicBCs(phi, lims):
    r"""
        Periodic only in x-direction - outflow still in y.
    """
    ilo, ihi, jlo, jhi = lims
    for i in range(2):
        phi[ilo:ihi, i] = phi[ilo:ihi,2]
        phi[ilo:ihi, -(i+1)] = phi[ilo:ihi,-3]

        phi[i,:] = phi[-4+i,:]
        phi[-(i+1),:] = phi[3-i,:]
    return phi


def dovis(phi, title, gridLims, iblims, n, t):
    """
    Do runtime visualization.
    """

    plt.clf()

    plt.rc("font", size=10)

    xmin, xmax, ymin, ymax = gridLims
    ilo, ihi, jlo, jhi = iblims
    levels = np.linspace(-0.4,1.0,8)

    img = plt.contour(np.transpose(phi[ilo:ihi, jlo:jhi]), levels,
                interpolation='nearest', origin="lower",
                extent=[xmin, xmax, ymin, ymax])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    plt.colorbar(img)

    plt.figtext(0.05,0.0125, "n = %d,    t = %10.5f" % (n, t))

    plt.draw()


if __name__ == "__main__":
    #testCircleEvolution()
    #testSquareEvolution()
    testSineEvolution()
