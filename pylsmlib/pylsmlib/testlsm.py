import numpy as np
import pythonisedfns
from pylsmlib import computeDistanceFunction
import matplotlib.pyplot as plt
import csv

__docformat__ = 'restructuredtext'


def testCircleEvolution():
    r"""
        Evolves an initially circular level set to see if it remains circular.
    """

    # create circle
    N = 50
    t = 0.
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    r = 0.3
    dx = 2.0 / (N - 1)
    dy = dx
    dt = 0.1
    phi = (X) ** 2 + (Y) ** 2 - r ** 2
    phi = computeDistanceFunction(phi, dx)
    title = "Circle evolution"

    # set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12, 9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
          [0, dx*(N-4), 0, dx*(N-4)], [2, N-3, 2, N-3], 0, 0)
    plt.show(block=False)

    nIt = 200
    sL0 = 0.0
    marksteinLength = 0.0
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    adVel = 0.1  # magnitude of advective velocity
    iblims = np.array([2, N-3, 2, N-3])
    lims = np.array([2, N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])

    # make velocities
    hyp = np.sqrt(X[:, :]**2 + Y[:, :]**2)
    u[:, :] = adVel * Y[:, :] / hyp[:, :]
    v[:, :] = adVel * X[:, :] / hyp[:, :]

    # fix signs
    u[:N/2, :] = -1. * np.fabs(u[:N/2, :])
    u[N/2:, :] = np.fabs(u[N/2:, :])
    v[:, :N/2] = -1. * np.fabs(v[:, :N/2])
    v[:, N/2:] = np.fabs(v[:, N/2:])

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, v,
                                               iblims, dx=dx, dy=dx)

        # calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u, v=v)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi, jlo:jhi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi] + Fx[ilo+1:ihi+1, jlo:jhi])/dx +
            (Fy[ilo:ihi, jlo:jhi] + Fy[ilo:ihi, jlo+1:jhi+1])/dy)

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
    N = 80
    X, Y = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N))
    t = 0.
    dx = 2.0 / (N - 1)
    dy = dx
    dt = 0.1
    phi = np.ones((N, N))
    phi[N/2-5:N/2+6, N/2-5:N/2+6] = -1.
    phi = computeDistanceFunction(phi, dx)
    title = "Square evolution"

    # set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12, 9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
          [0, dx*(N-4), 0, dx*(N-4)], [2, N-3, 2, N-3], 0, 0)
    plt.show(block=False)

    nIt = 250
    sL0 = 0.0
    marksteinLength = 0.0
    adVel = 0.05  # magnitude of advective velocity
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    iblims = np.array([2, N-3, 2, N-3])
    lims = np.array([2, N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])

    # make velocities
    # rhs
    mask = (Y > 0.) * (Y > np.abs(X))
    u[mask] = adVel
    # lhs
    mask = (Y < 0.) * (np.fabs(Y) > np.abs(X))
    u[mask] = -1.*adVel
    # top
    mask = (X > 0.) * (X > np.abs(Y))
    v[mask] = adVel
    # bottom
    mask = (X < 0.) * (np.fabs(X) > np.abs(Y))
    v[mask] = -1.*adVel

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, v,
                                               iblims, dx=dx, dy=dx)

        # calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u, v=v)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi, jlo:jhi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi] + Fx[ilo+1:ihi+1, jlo:jhi])/dx +
            (Fy[ilo:ihi, jlo:jhi] + Fy[ilo:ihi, jlo+1:jhi+1])/dy)

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

    # create sine
    N = 85
    sinewidth = np.rint(N/10)
    sinewidth = sinewidth.astype(int)
    t = 0.
    dx = 2.0 / (N - 5)
    dy = dx
    dt = 0.1
    phi = np.ones((N, N))
    phi[N/2-sinewidth:N/2+sinewidth+1, :] = -1.
    phi = computeDistanceFunction(phi, dx)
    title = "Periodic evolution"
    toCSV = False

    # set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12, 9), dpi=100, facecolor='w')
    dovis(phi, title, [0, dx*(N-5), 0, dx*(N-5)], [2, N-3, 2, N-3], 0, 0)
    plt.show(block=False)

    nIt = 150
    sL0 = 0.08
    marksteinLength = 0.05
    u = np.ones_like(phi)*0.08
    iblims = np.array([2, N-3, 2, N-3])
    lims = np.array([2, N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-5), 0, dx*(N-5)])

    if toCSV:
        headers = ['t', 'xcoord', 'ycoord', 'phi']
        filepath = \
            '/home/alice/Documents/Dropbox/LSMLIB/pylsmlib/pylsmlib/fire/'
        xs = [dx*n for n in range(N-4)]
        ys = [dy*n for n in range(N-4)]

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, u,
                                               iblims, dx=dx, dy=dx)

        # calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u, v=u)
        Fx = enforcePeriodicBCs(Fx, lims)
        Fy = enforcePeriodicBCs(Fy, lims)

        phi[ilo:ihi, jlo:jhi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi] + Fx[ilo+1:ihi+1, jlo:jhi])/dx +
            (Fy[ilo:ihi, jlo:jhi] + Fy[ilo:ihi, jlo+1:jhi+1])/dy)

        t += dt
        # enforce periodic boundary conditions
        phi[:, :] = enforcePeriodicBCs(phi[:, :], lims)

        # plotting
        dovis(phi, title, gridlims, iblims, n, t)
        plt.show(block=False)

        # save to file
        if toCSV:
            # remember to ignore ghosts
            rows = [(t, xs[i], ys[j], phi[i, j]) for j in range(N-4)
                    for i in range(N-4)]

            with open(filepath + 'sine' + str(n) + '.csv', 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)

        # reinitialise
        phi[:, :] = computeDistanceFunction(phi[:, :], dx)

    return


def testSphereEvolution():
    r"""
        Evolves an initially spherical set to see if it remains spherical.
    """

    # create circle
    N = 20
    t = 0.
    X, Y, Z = np.meshgrid(np.linspace(-1, 1, N), np.linspace(-1, 1, N),
                          np.linspace(-1, 1, N))
    r = 0.3
    dx = 2.0 / (N - 1)
    dy = dx
    dz = dx
    dt = 0.1
    phi = (X) ** 2 + (Y) ** 2 + (Z) ** 2 - r ** 2
    phi = computeDistanceFunction(phi, dx)
    title = "Sphere evolution"
    toCSV = False

    # set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12, 9), dpi=100, facecolor='w')
    dovis(np.transpose(phi[N/2, :, :]), title,
          [0, dx*(N-4), 0, dx*(N-4)], [2, N-3, 2, N-3], 0, 0)
    plt.show(block=False)

    nIt = 100
    sL0 = 0.2
    marksteinLength = 0.05
    u = 0.1 * np.zeros_like(phi)
    iblims = np.array([2, N-3, 2, N-3, 2, N-3])
    lims = np.array([2, N-2, 2, N-2, 2, N-2])
    ilo, ihi, jlo, jhi, klo, khi = lims
    # gridlims = np.array([0, dx*(N-4), 0, dx*(N-4), 0, dx*(N-4)])

    if toCSV:
        headers = ['t', 'xcoord', 'ycoord', 'zcoord', 'phi']
        filepath = \
            '/home/alice/Documents/Dropbox/LSMLIB/pylsmlib/pylsmlib/fire/'
        xs = [dx*n for n in range(N-4)]
        ys = [dy*n for n in range(N-4)]
        zs = [dz*n for n in range(N-4)]

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed(phi, sL0, marksteinLength, u, u,
                                             u, iblims, dx=dx, dy=dx, dz=dz)

        # calculate fluxes
        Fx, Fy, Fz = findFluxes3d(phi, sL, lims, dx=dx, dy=dy, dz=dz, u=u,
                                  v=u, w=u)
        Fx = enforceOutflowBCs3d(Fx, lims)
        Fy = enforceOutflowBCs3d(Fy, lims)
        Fz = enforceOutflowBCs3d(Fz, lims)

        phi[ilo:ihi, jlo:jhi, klo:khi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi, klo:khi] +
             Fx[ilo+1:ihi+1, jlo:jhi, klo:khi]) / dx +
            (Fy[ilo:ihi, jlo:jhi, klo:khi] +
             Fy[ilo:ihi, jlo+1:jhi+1, klo:khi]) / dy +
            (Fz[ilo:ihi, jlo:jhi, klo:khi] +
             Fz[ilo:ihi, jlo:jhi, klo+1:khi+1]) / dx)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs3d(phi, lims)

        # plotting
        dovis(np.transpose(phi[N/2, :, :]), title,
              [0, dx*(N-4), 0, dx*(N-4)], [2, N-3, 2, N-3], n, t)
        plt.show(block=False)

        # save to file
        if toCSV:
            # remember to ignore ghosts
            rows = [(t, xs[i], ys[j], zs[k], phi[i, j, k]) for k in range(N-4)
                    for j in range(N-4) for i in range(N-4)]

            with open(filepath + 'sphere' + str(n) + '.csv', 'w') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(headers)
                f_csv.writerows(rows)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

    return


def testVortexEvolution():
    r"""
        Evolves a circle in a vortex. The vortex is initialised with the stream
        function

        ..math::

            \Psi = \frac{1}{\pi}\sin^2(\pi x)\sin^2(\pi y),

        where the velocities are then given by
        :math:`u = \partial\Psi/\partial y`,
        :math:`v = -\partial\Psi/\partial x`.

        The vortex reverses direction after a certain number of timesteps to see
        how close the system returns to the initial conditions.

    """

    # create circle
    N = 80
    t = 0.
    X, Y = np.meshgrid(np.linspace(0., 1., N), np.linspace(0., 1., N))
    r = 0.15
    dx = 1.0 / (N - 1.)
    dy = dx
    dt = 0.05
    phi = (X-0.5) ** 2 + (Y-0.75) ** 2 - r ** 2
    phi = computeDistanceFunction(phi, dx)
    title = "Vortex evolution"

    # set up plot
    plt.ion()
    plt.figure(num=1, figsize=(12, 9), dpi=100, facecolor='w')
    dovis(np.transpose(phi), title,
          [0, dx*(N-4), 0, dx*(N-4)], [2, N-3, 2, N-3], 0, 0)
    plt.show(block=False)

    nIt = 200
    sL0 = 0.0
    marksteinLength = 0.0
    u = np.zeros_like(phi)
    v = np.zeros_like(phi)
    adVel = 0.1  # magnitude of advective velocity
    iblims = np.array([2, N-3, 2, N-3])
    lims = np.array([2, N-2, 2, N-2])
    ilo, ihi, jlo, jhi = lims
    gridlims = np.array([0, dx*(N-4), 0, dx*(N-4)])

    # make velocities
    u[:, :] = adVel * np.sin(Y[:, :] * np.pi)**2 * np.sin(2. * X[:,:] * np.pi)
    v[:, :] = -adVel * np.sin(2. * Y[:, :] * np.pi) * np.sin(X[:,:] * np.pi)**2

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, v,
                                               iblims, dx=dx, dy=dx)

        # calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u, v=v)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi, jlo:jhi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi] + Fx[ilo+1:ihi+1, jlo:jhi])/dx +
            (Fy[ilo:ihi, jlo:jhi] + Fy[ilo:ihi, jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, lims)

        # plotting
        dovis(np.transpose(phi), title, gridlims, iblims, n, t)
        #streamplot(np.linspace(-0.5, 0.5, N), u, v, title)
        plt.show(block=False)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

    # now reverse the direction of the vortex and rewind
    u[:, :] = -u[:, :]
    v[:, :] = -v[:, :]

    for n in range(nIt):
        # flame speed
        sL = pythonisedfns.laminarFlameSpeed2d(phi, sL0, marksteinLength, u, v,
                                               iblims, dx=dx, dy=dx)

        # calculate fluxes
        Fx, Fy = findFluxes(phi, sL, lims, dx=dx, dy=dy, u=u, v=v)
        Fx = enforceOutflowBCs(Fx, lims)
        Fy = enforceOutflowBCs(Fy, lims)

        phi[ilo:ihi, jlo:jhi] -= 0.5 * dt * (
            (Fx[ilo:ihi, jlo:jhi] + Fx[ilo+1:ihi+1, jlo:jhi])/dx +
            (Fy[ilo:ihi, jlo:jhi] + Fy[ilo:ihi, jlo+1:jhi+1])/dy)

        t += dt
        # enforce outflow boundary conditions
        phi = enforceOutflowBCs(phi, lims)

        # plotting
        dovis(np.transpose(phi), title, gridlims, iblims, n+nIt, t)
        #streamplot(np.linspace(-0.5, 0.5, N), u, v, title)
        plt.show(block=False)

        # reinitialise
        phi = computeDistanceFunction(phi, dx)

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
    norm_x, norm_y = pythonisedfns.signedUnitNormal2d(phi, phi_x, phi_y,
                                                      dx=dx, dy=dy)

    # normalise norms some more
    norm_x[:, :] /= np.sqrt(norm_x[:, :]**2 + norm_y[:, :]**2)
    norm_y[:, :] /= np.sqrt(norm_x[:, :]**2 + norm_y[:, :]**2)

    # add on laminar flame speed contribution
    u[:, :] += norm_x[:, :] * sL[:, :]
    v[:, :] += norm_y[:, :] * sL[:, :]

    Fx[ilo:ihi, jlo:jhi] = u[ilo:ihi, jlo:jhi] * phi[ilo:ihi, jlo:jhi] - \
        u[ilo-1:ihi-1, jlo:jhi] * phi[ilo-1:ihi-1, jlo:jhi]
    Fy[ilo:ihi, jlo:jhi] = v[ilo:ihi, jlo:jhi] * phi[ilo:ihi, jlo:jhi] - \
        v[ilo:ihi, jlo-1:jhi-1] * phi[ilo:ihi, jlo-1:jhi-1]

    return Fx, Fy


def findFluxes3d(phi, sL, lims, dx=1., dy=1., dz=1., u=None, v=None, w=None):
    r"""
        Find the fluxes.
    """

    ilo, ihi, jlo, jhi, klo, khi = lims
    Fx = np.zeros_like(phi)
    Fy = np.zeros_like(phi)
    Fz = np.zeros_like(phi)

    if u is None:
        u = np.zeros_like(phi)
    if v is None:
        v = np.zeros_like(phi)
    if w is None:
        w = np.zeros_like(phi)

    phi_x, phi_y, phi_z = pythonisedfns.gradPhi(phi, dx=dx, dy=dy, dz=dz)
    norm_x, norm_y, norm_z = pythonisedfns.signedUnitNormal(phi, phi_x, phi_y,
                                                            phi_z, dx=dx,
                                                            dy=dy, dz=dz)

    # normalise norms some more
    norm_x[:, :, :] /= np.sqrt(norm_x[:]**2 + norm_y[:]**2 + norm_z[:]**2)
    norm_y[:, :, :] /= np.sqrt(norm_x[:]**2 + norm_y[:]**2 + norm_z[:]**2)
    norm_z[:, :, :] /= np.sqrt(norm_x[:]**2 + norm_y[:]**2 + norm_z[:]**2)

    # add on laminar flame speed contribution
    u[:, :, :] += norm_x[:, :, :] * sL[:, :, :]
    v[:, :, :] += norm_y[:, :, :] * sL[:, :, :]
    w[:, :, :] += norm_z[:, :, :] * sL[:, :, :]

    Fx[ilo:ihi, jlo:jhi, klo:khi] = u[ilo:ihi, jlo:jhi, klo:khi] * \
        phi[ilo:ihi, jlo:jhi, klo:khi] - \
        u[ilo-1:ihi-1, jlo:jhi, klo:khi] * phi[ilo-1:ihi-1, jlo:jhi, klo:khi]
    Fy[ilo:ihi, jlo:jhi, klo:khi] = v[ilo:ihi, jlo:jhi, klo:khi] * \
        phi[ilo:ihi, jlo:jhi, klo:khi] - \
        v[ilo:ihi, jlo-1:jhi-1, klo:khi] * phi[ilo:ihi, jlo-1:jhi-1, klo:khi]
    Fz[ilo:ihi, jlo:jhi, klo:khi] = w[ilo:ihi, jlo:jhi, klo:khi] * \
        phi[ilo:ihi, jlo:jhi, klo:khi] - \
        w[ilo:ihi, jlo:jhi, klo-1:khi-1] * phi[ilo:ihi, jlo:jhi, klo-1:khi-1]

    return Fx, Fy, Fz


def enforceOutflowBCs(phi, lims):
    r"""
        Just copy across outermost cells of inner box
    """
    _, _, jlo, jhi = lims
    for i in range(2):
        phi[i, jlo:jhi] = phi[2, jlo:jhi]
        phi[-(i+1), jlo:jhi] = phi[-3, jlo:jhi]
        phi[:, i] = phi[:, 2]
        phi[:, -(i+1)] = phi[:, -3]
    return phi


def enforceOutflowBCs3d(phi, lims):
    r"""
        Just copy across outermost cells of inner box
    """
    _, _, jlo, jhi, klo, khi = lims
    for i in range(2):
        phi[i, jlo:jhi, klo:khi] = phi[2, jlo:jhi, klo:khi]
        phi[-(i+1), jlo:jhi, klo:khi] = phi[-3, jlo:jhi, klo:khi]
        phi[:, i, klo:khi] = phi[:, 2, klo:khi]
        phi[:, -(i+1), klo:khi] = phi[:, -3, klo:khi]
        phi[:, :, i] = phi[:, :, 2]
        phi[:, :, -(i+1)] = phi[:, :, -3]
    return phi


def enforcePeriodicBCs(phi, lims):
    r"""
        Periodic only in x-direction - outflow still in y.
    """
    ilo, ihi, _, _ = lims
    for i in range(2):
        phi[ilo:ihi, i] = phi[ilo:ihi, 2]
        phi[ilo:ihi, -(i+1)] = phi[ilo:ihi, -3]

        phi[i, :] = phi[-4+i, :]
        phi[-(i+1), :] = phi[3-i, :]
    return phi


def dovis(phi, title, gridLims, iblims, n, t):
    """
    Do runtime visualization.
    """

    plt.clf()

    plt.rc("font", size=10)

    xmin, xmax, ymin, ymax = gridLims
    ilo, ihi, jlo, jhi = iblims
    levels = np.linspace(-0.4, 1.0, 8)

    img = plt.contour(np.transpose(phi[ilo:ihi, jlo:jhi]), levels,
                      interpolation='nearest', origin="lower",
                      extent=[xmin, xmax, ymin, ymax])

    # img = plt.imshow(np.transpose(phi[ilo:ihi, jlo:jhi]),
    #            interpolation='bilinear', origin="lower",
    #            extent=[xmin, xmax, ymin, ymax], vmin=-0.4, vmax=1.)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    plt.colorbar(img)

    plt.figtext(0.05, 0.0125, "n = %d,    t = %10.5f" % (n, t))

    plt.draw()
    # if n >120:
    #    plt.savefig("fire/sinePlot" + str(n) + ".png")


def streamplot(x, u, v, title):
    """
    Plot velocity streamlines.
    """
    plt.clf()

    plt.rc("font", size=10)

    img = plt.streamplot(x, x, np.transpose(u), np.transpose(v))

    #img = plt.imshow(np.transpose(u**2 + v**2),
    #            interpolation='bilinear', origin="lower")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)

    #plt.colorbar(img)

    #plt.figtext(0.05, 0.0125, "n = %d,    t = %10.5f" % (n, t))

    plt.draw()



if __name__ == "__main__":
    # testCircleEvolution()
    # testSquareEvolution()
    # testSineEvolution()
    testVortexEvolution()
    # testSphereEvolution()
