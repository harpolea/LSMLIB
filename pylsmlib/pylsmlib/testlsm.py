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
    sL0 = 1.
    marksteinLength = 0.01
    u = np.zeros_like(phi)
    iblims = np.array([2,N-3, 2, N-3])
    ilo, ihi, jlo, jhi = np.array([2,N-2, 2, N-2])
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        phi_x, phi_y = pythonisedfns.gradPhi2d(phi, dx=dx, dy=dy)
        norm_x, norm_y = pythonisedfns.signedUnitNormal2d(phi, phi_x, phi_y, dx=dx, dy=dy)

        #calculate fluxes
        Fx = np.zeros_like(phi)
        Fy = np.zeros_like(phi)

        phi[ilo:ihi,jlo:jhi] += dt * ( \
            (Fx[ilo:ihi,jlo:jhi] - Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] - Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, [ilo, ihi, jlo, jhi])

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
    sL0 = 1.
    marksteinLength = 0.01
    u = np.zeros_like(phi)
    iblims = np.array([2,N-3, 2, N-3])
    ilo, ihi, jlo, jhi = np.array([2,N-2, 2, N-2])
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        phi_x, phi_y = pythonisedfns.gradPhi2d(phi, dx=dx, dy=dy)
        norm_x, norm_y = pythonisedfns.signedUnitNormal2d(phi, phi_x, phi_y, dx=dx, dy=dy)

        #calculate fluxes
        Fx = np.zeros_like(phi)
        Fy = np.zeros_like(phi)

        phi[ilo:ihi,jlo:jhi] += dt * ( \
            (Fx[ilo:ihi,jlo:jhi] - Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] - Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, [ilo, ihi, jlo, jhi])

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
    dt    = 0.001
    phi   = np.ones((N,N))
    phi[N/2-9:N/2+10,:] = -1.
    phi   = computeDistanceFunction(phi, dx)
    title = "Periodic evolution"

    #set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12,9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
                [0, dx*(N-4), 0, dx*(N-4)], [2,N-3, 2, N-3], 0,0)
    plt.show(block=False)

    nIt = 50
    sL0 = 1.
    marksteinLength = 0.01
    u = np.zeros_like(phi)
    iblims = np.array([2,N-3, 2, N-3])
    ilo, ihi, jlo, jhi = np.array([2,N-2, 2, N-2])
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])


    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u, iblims, dx=dx, dy=dx)

        phi_x, phi_y = pythonisedfns.gradPhi2d(phi, dx=dx, dy=dy)
        norm_x, norm_y = pythonisedfns.signedUnitNormal2d(phi, phi_x, phi_y, dx=dx, dy=dy)

        #calculate fluxes
        Fx = np.zeros_like(phi)
        Fy = np.zeros_like(phi)

        phi[ilo:ihi,jlo:jhi] += dt * ( \
            (Fx[ilo:ihi,jlo:jhi] - Fx[ilo+1:ihi+1,jlo:jhi])/dx + \
            (Fy[ilo:ihi,jlo:jhi] - Fy[ilo:ihi,jlo+1:jhi+1])/dy)

        t += dt
        # enforce periodic boundary conditions
        phi = enforcePeriodicBCs(phi, [ilo, ihi, jlo, jhi])

        # plotting
        dovis(phi, title, gridlims, iblims, n, t)
        plt.show(block=False)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

    return


def findFluxes(phi):
    pass


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
        #phi[i,jlo:jhi] = phi[2,jlo:jhi]
        #phi[-(i+1),jlo:jhi] = phi[-3,jlo:jhi]

        #phi[:,i] = phi[:,-4+i]
        #phi[:,-(i+1)] = phi[:,3-i]

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

    img = plt.contour(np.transpose(phi), interpolation='nearest', origin="lower", extent=[xmin, xmax, ymin, ymax])

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    plt.colorbar(img)

    plt.figtext(0.05,0.0125, "n = %d,    t = %10.5f" % (n, t))

    plt.draw()



if __name__ == "__main__":
    #testCircleEvolution()
    testSquareEvolution()
    testSineEvolution()
