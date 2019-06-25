import numpy as np
import cmath


def num_params(n):
    """ decorator to store number of parameters for an element """
    def decorator(func):
        def wrapper(p, f):
            typeChecker(p, f, func.__name__, n)
            return func(p, f)

        wrapper.num_params = n
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__

        return wrapper
    return decorator


def s(series):
    """ sums elements in series

    Notes
    ---------
    .. math::
        Z = Z_1 + Z_2 + ... + Z_n

    """
    z = len(series[0])*[0 + 0*1j]
    for elem in series:
        z += elem
    return z


def p(parallel):
    """ adds elements in parallel

    Notes
    ---------
    .. math::

        Z = \\frac{1}{\\frac{1}{Z_1} + \\frac{1}{Z_2} + ... + \\frac{1}{Z_n}}

     """
    z = len(parallel[0])*[0 + 0*1j]
    for elem in parallel:
        z += 1/elem
    return 1/z


@num_params(1)
def R(p, f):
    """ defines a resistor

    Notes
    ---------
    .. math::

        Z = R

    """
    return np.array(len(f)*[p[0]])


@num_params(1)
def C(p, f):
    """ defines a capacitor

    .. math::

        Z = \\frac{1}{C \\times j 2 \\pi f}

     """
    omega = 2*np.pi*np.array(f)
    return 1.0/(p[0]*1j*omega)


@num_params(1)
def L(p, f):
    """ defines an inductor

    .. math::

        Z = L \\times j 2 \\pi f

     """
    omega = 2*np.pi*np.array(f)
    return p[0]*1j*omega


@num_params(2)
def W(p, f):
    """ defines a blocked boundary Finite-length Warburg Element

    Notes
    ---------
    .. math::
        Z = \\frac{R}{\\sqrt{ T \\times j 2 \\pi f}}
        \\coth{\\sqrt{T \\times j 2 \\pi f }}

    where :math:`R` = p[0] (Ohms) and
    :math:`T` = p[1] (sec) = :math:`\\frac{L^2}{D}`

    """
    omega = 2*np.pi*np.array(f)
    Zw = np.vectorize(lambda y: p[0]/(np.sqrt(p[1]*1j*y) *
                                      cmath.tanh(np.sqrt(p[1]*1j*y))))
    return Zw(omega)


@num_params(1)
def A(p, f):
    """ defines a semi-infinite Warburg element

    Notes
    -----
    .. math::

        Z = \\frac{A_W}{\\sqrt{ 2 \\pi f}} (1-j)
    """
    omega = 2*np.pi*np.array(f)
    Aw = p[0]
    Zw = Aw*(1-1j)/np.sqrt(omega)
    return Zw


@num_params(2)
def E(p, f):
    """ defines a constant phase element

    Notes
    -----
    .. math::

        Z = \\frac{1}{Q \\times (j 2 \\pi f)^\\alpha}

    where :math:`Q` = p[0] and :math:`\\alpha` = p[1].
    """
    omega = 2*np.pi*np.array(f)
    Q, alpha = p
    return 1.0/(Q*(1j*omega)**alpha)


@num_params(2)
def G(p, f):
    """ defines a Gerischer Element

    Notes
    ---------
    .. math::

        Z = \\frac{1}{Y \\times \\sqrt{K + j 2 \\pi f }}

     """
    omega = 2*np.pi*np.array(f)
    Z0, k = p
    return Z0/np.sqrt(k + 1j*omega)


@num_params(2)
def K(p, f):
    """ An RC element for use in lin-KK model

    Notes
    -----
    .. math::

        Z = \\frac{R}{1 + j \\omega \\tau_k}

    """
    omega = np.array(f)
    return p[0]/(1 + 1j*omega*p[1])


@num_params(4)
def T(p, f):
    """ A macrohomogeneous porous electrode model from Paasch et al. [1]

    Notes
    -----
    .. math::

        Z = A\\frac{\\coth{\\beta}}{\\beta} + B\\frac{1}{\\beta\\sinh{\\beta}}

    where

    .. math::

        A = d\\frac{\\rho_1^2 + \\rho_2^2}{\\rho_1 + \\rho_2} \\quad
        B = d\\frac{2 \\rho_1 \\rho_2}{\\rho_1 + \\rho_2}

    and

    .. math::
        \\beta = (a + j \\omega b)^{1/2} \\quad
        a = \\frac{k d^2}{K} \\quad b = \\frac{d^2}{K}


    [1] G. Paasch, K. Micka, and P. Gersdorf,
    Electrochimica Acta, 38, 2653–2662 (1993)
    `doi: 10.1016/0013-4686(93)85083-B
    <https://doi.org/10.1016/0013-4686(93)85083-B>`_.
    """

    omega = 2*np.pi*np.array(f)
    A, B, a, b = p
    beta = (a + 1j*omega*b)**(1/2)

    sinh = []
    for x in beta:
        if x < 100:
            sinh.append(np.sinh(x))
        else:
            sinh.append(1e10)

    return A/(beta*np.tanh(beta)) + B/(beta*np.array(sinh))

#### new element
    
@num_params(15)
def V(p, f):
    """ Machrohomogeneous porous electrode for RFB [1]

    Notes
    -----
    .. math::git 




    [1] A. M. Pezeshki et al.,
    Electrochimica Acta, 229, 261-270 (2017)
    `doi: 10.1016/j.electacta.2017.01.056'
    """

    omega = 2*np.pi*np.array(f)
    A,b,C_dl,c_o,c_r,D_o,D_r,rho_e,rho_i,T,A_t,i_0,q,a,f = p
    
    ## define R and F
    R=8.31446261815324 #J K-1 mol-1
    F=96485.33289 # C mol-1 
    
    sinh,cosh = [],[]
    for x in a*np.sqrt((1j*omega)/D_r):
        if x < 100:
            sinh.append(np.sinh(x))
            cosh.append(np.cosh(x))
        else:
            sinh.append(1e10)
            cosh.append(1e10)			
    tanhDR=np.divide(np.asarray(sinh),np.asarray(cosh))

    sinh,cosh = [],[]
    for x in a*np.sqrt((1j*omega)/D_o):
        if x < 100:
            sinh.append(np.sinh(x))
            cosh.append(np.cosh(x))
        else:
            sinh.append(1e10)
            cosh.append(1e10)			
    tanhDO=np.divide(np.asarray(sinh),np.asarray(cosh))
    #defining Z_a
    #DDR=  ((R*T*a)/(omega*F**2*c_r*D_r)) * ((np.tanh(a*np.sqrt((1j*omega)/D_r)))/(a*np.sqrt((1j*omega)/D_r)))
    #DDO = ((R*T*a)/(omega*F**2*c_o*D_o))*((np.tanh(a*np.sqrt((1j*omega)/D_o)))/(a*np.sqrt((1j*omega)/D_o)))   
    #u=(R*T)/(F*i_0) + DDR + DDO
    #Za=1/(1/u+ C_dl*(1j*omega)**q)
    
    #defining Z_a
    DDR=  ((R*T*a)/(f*F**2*c_r*D_r)) * (tanhDR/(a*np.sqrt((1j*omega)/D_r)))
    DDO = ((R*T*a)/(f*F**2*c_o*D_o))*(tanhDO/(a*np.sqrt((1j*omega)/D_o)))     
    u=(R*T)/(F*i_0) + DDR + DDO
    Za=1/(1/u+ C_dl*(1j*omega)**q)
    
    #defining chi   
    chi=np.sqrt( (  (rho_e+rho_i)*b*A_t) / (A*Za)  ) 
    
    sinh = []
    for x in chi:
        if x < 100:
            sinh.append(np.sinh(x))
        else:
            sinh.append(1e10)	
    cosh = []
    for x in chi:
        if x < 100:
            cosh.append(np.cosh(x))
        else:
            cosh.append(1e10)	
     
    coth=np.divide(np.asarray(cosh),np.asarray(sinh))

    #defining Zp   
    b1=b*((rho_e**2+rho_i**2)/(rho_e+rho_i))* ( (coth)/(chi) )
    b2=b*((rho_e*rho_i*2)/(rho_e+rho_i))* ( (1)/(chi*sinh) )
    b3=b*((rho_e*rho_i)/(rho_e+rho_i))
    Zp=(b1+b2+b3)/A

    #defining Zp   
    #b1=b*((rho_e**2+rho_i**2)/(rho_e+rho_i))* ( (1/ np.tanh(chi))/(chi) )
    #b2=b*((rho_e*rho_i*2)/(rho_e+rho_i))* ( (1)/(chi*np.sinh(chi)) )
    #b3=b*((rho_e*rho_i)/(rho_e+rho_i))
    #Zp=(b1+b2+b3)/A
    

    return Zp


def typeChecker(p, f, name, length):
    assert isinstance(p, list), \
        'in {}, input must be of type list'.format(name)
    for i in p:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, p)
    for i in f:
        assert isinstance(i, (float, int, np.int32, np.float64)), \
            'in {}, value {} in {} is not a number'.format(name, i, f)
    assert len(p) == length, \
        'in {}, input list must be length {}'.format(name, length)
    return
