from astropy.coordinates import Angle, SkyCoord, get_sun
from astropy.constants import G, M_earth, R_earth
from astropy.time import Time
from astropy import units as u

import numpy as np

def apparent_to_absolute(magapp, jd, ra, dec):
    """ Compute absolute magnitude from apparent magnitude
    
    Parameters
    ----------
    magapp: array
        Apparent magnitudes of alerts
    jd: array
        Times (JD) of alerts
    ra: array
        RA of alerts (deg)
    dec: array
        Dec of alerts (deg)
        
    Returns
    ----------
    magabs: array
        Absolute magnitude of alerts
    """
    sun_pos = get_sun(Time(jd, format='jd'))
    sun_dists = sun_pos.separation(SkyCoord(ra, dec, unit='deg')).rad
    
    denominator = np.sin(sun_dists) + (np.pi - sun_dists) * np.cos(sun_dists)
    magabs = magapp - 5. / 2. * np.log10(np.pi / denominator)
    return magabs, sun_dists

def fake_lambertian_size(magabs, msun, distance, albedo):
    """ Try to estimate the size of an object using Lambertian formula
    
    Parameters
    ----------
    magabs: array
        Absolute magnitudes of alerts
    msun: float
        Apparent magnitude of the Sun
    distance: array
        Distance between the object and us [km]
    albedo: float
        Albedo - take 0.175 (2012.12549)
    """
    return 10**((msun - magabs)/5.) * distance * np.sqrt(6/albedo)

def fake_lambertian_size_apparent(magapp, msun, distance, sun_dists, albedo):
    """ Try to estimate the size of an object using Lambertian formula
    
    Parameters
    ----------
    magabs: array
        Absolute magnitudes of alerts
    msun: float
        Apparent magnitude of the Sun
    distance: array
        Distance between the object and us [km]
    albedo: float
        Albedo - take 0.175 (2012.12549)
    """
    denominator = np.sin(sun_dists) + (np.pi - sun_dists) * np.cos(sun_dists)
    return 10**((msun - magapp)/5.) * distance * np.sqrt(1/albedo) * np.sqrt(6 * np.pi / denominator)

def get_semi_major_axis(period):
    """ Compute semi major axis in km
    """
    mu = 3.986 * 10**5
    period_seconds = period * 3600.
    a = ((period_seconds / 2. / np.pi)**2 * mu)**(1./3.)
    return a

def get_period(velocity):
    """ Compute rotation period over Earth of an object given its velocity
    
    Parameters
    ----------
    velocity: float or array
        Velocity of object(s) in deg/`time`
        
    Returns
    ----------
    period: float or array
        Period in `time`
    """
    return 360. / velocity

def get_velocity_bystep(data, single_exposure_time = 30., min_alert_per_exposure = 1):
    """ Estimate the velocity of the tracklet in deg/hour
    
    If the tracklet spans several exposures, the user can discard exposures
    with not enough points (min_alert_per_exposure). Exposure with less than 2 points 
    are usually not reliable (even if the entire tracklet has many points). Note that the 
    total tracklet must still have at least 5 alerts to be processed (otherwise it is discarded).
    
    To compute the velocity, we integrate the distance between subsequent measurements:
    v = sum_i[x(i+1) - x(i)] / dt, i=alert
    where dt is the time between jdstart(first exposure) & jdstart(last exposure) + last exposure time
    
    Parameters
    ----------
    data: Table
        table containing all the data for one tracklet
    single_exposure_time: float
        ZTF exposure time. Default is 30 seconds.
    min_alert_per_exposure: int
        Minimum alerts per exposure to consider the exposure data as valid. Exposure 
        with less data points will be discarded from the analysis. If the total 
        number of points in the tracklets becomes less than 5, the entire tracklet
        is discarded.
        
    Returns
    ----------
    velocity: float or NaN
        Velocity in deg/hour. Invalid tracklets return NaN.
    """
    # Initialise the trajectory
    length = 0.0
    
    # Get unique exposure
    jd_unique = np.unique(data['jd']).data
    n_exposure = len(jd_unique)
    
    # Treat the case where tracklets span several exposures
    mask_exposure = np.ones_like(data['jd'], dtype=bool)

    if n_exposure > 1:
        # remove exposures with nalert < min_alert_per_exposure
        for k in jd_unique:
            mask_ = data['jd'] == k
            l = np.sum(mask_)
            if l < min_alert_per_exposure:
                mask_exposure[mask_] = False
        
        # Discard if there are not enough alerts left (we want a total of 5 at least)
        if np.sum(mask_exposure) < 5:
            return np.nan
        
        # Compute the time lapse between the start of the first and last exposure (0 if the same exposure)
        delta_jd_second = (
            np.max(data['jd'][mask_exposure]) - np.min(data['jd'][mask_exposure])
        ) * 24. * 3600.

        # add the last exposure time
        single_exposure_time = delta_jd_second + 30.
    
    # Sort data, and integrate the trajectory
    data.sort('dec')
    for i in range(len(data[mask_exposure]) - 1):
        first = SkyCoord(data['ra'][mask_exposure][i], data['dec'][mask_exposure][i], unit='deg')
        last = SkyCoord(data['ra'][mask_exposure][i+1], data['dec'][mask_exposure][i+1], unit='deg')
        length += first.separation(last).degree
        
    # return the velocity in deg/hour
    return length / (single_exposure_time / 3600.)

def wrap_angle(rad):
    ang = Angle(rad, u.radian).degree
    if ang > 90.:
        ang = 180 - ang
    return ang
    
    
def get_inclination_bystep(data, distances, single_exposure_time = 30., min_alert_per_exposure = 1):
    """ compute inclination in the range [0, 90] degree
    
    Parameters
    ----------
    data: Table
        table containing all the data for one tracklet
    single_exposure_time: float
        ZTF exposure time. Default is 30 seconds.
    min_alert_per_exposure: int
        Minimum alerts per exposure to consider the exposure data as valid. Exposure 
        with less data points will be discarded from the analysis. If the total 
        number of points in the tracklets becomes less than 5, the entire tracklet
        is discarded.
        
    Returns
    ----------
    inclination: float or NaN
        Mean inclination in degree. Invalid tracklets return NaN.
    """
    data.sort('dec')

    # Initialise the trajectory
    length = 0.0
    
    # Get unique exposure
    jd_unique = np.unique(data['jd']).data
    n_exposure = len(jd_unique)
    
    # Treat the case where tracklets span several exposures
    mask_exposure = np.ones_like(data['jd'], dtype=bool)
    if np.sum(mask_exposure) < 5:
        return np.nan, np.nan

    if n_exposure > 1:
        # remove exposures with nalert < min_alert_per_exposure
        for k in jd_unique:
            mask_ = data['jd'] == k
            l = np.sum(mask_)
            if l < min_alert_per_exposure:
                mask_exposure[mask_] = False
        
        # Discard if there are not enough alerts left (we want a total of 5 at least)
        if np.sum(mask_exposure) < 5:
            return np.nan, np.nan
        
        # Compute the time lapse between the start of the first and last exposure (0 if the same exposure)
        delta_jd_second = (
            np.max(data['jd'][mask_exposure]) - np.min(data['jd'][mask_exposure])
        ) * 24. * 3600.

        # add the last exposure time
        single_exposure_time = delta_jd_second + 30.
    
    # Sort data, and integrate the trajectory
#     data.sort('ra')
    coords = SkyCoord(
        data[mask_exposure]['ra']*u.deg, 
        data[mask_exposure]['dec']*u.deg, 
        distance=distances[mask_exposure]*u.km, 
    ).cartesian
    xyz = coords.xyz.value.T
    dist_total = np.sum([(i - j)**2 for i, j in zip(xyz[-1], xyz[0])])
    vs = []
    for i in range(len(data[mask_exposure]) - 1):
        x1, y1, z1 = xyz[i]
        x2, y2, z2 = xyz[i+1]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        dt = dist / dist_total * single_exposure_time
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vz = (z2 - z1) / dt
        vs.append([vx, vy, vz])
    
    try:
        h_bar = np.cross(xyz[0:-1],vs)
    except ValueError:
        
        print(np.shape(xyz[0:-1]), xyz[0:-1])
        print(np.shape(vs), vs)
    h = np.linalg.norm(h_bar, axis=1)
    incl = np.array([np.arccos(h_bar_[2]/h_) for h_bar_, h_ in zip(h_bar, h)])

    # wrap between 0 and 90 degrees
    mean_inclination = np.mean([wrap_angle(i) for i in incl])
    err_inclination = np.std([wrap_angle(i) for i in incl])
#     # rad to degree conversion
#     if np.all(incl >= np.pi):
#         mean_inclination = np.mean(incl) * 180. / np.pi
#         err_inclination = np.std(incl) * 180. / np.pi
#     elif np.any(incl >= np.pi):
#         # wrap between 0 and 90 degrees
#         mean_inclination = np.mean([wrap_angle(i) for i in incl])
#         err_inclination = np.std([wrap_angle(i) for i in incl])
#     else:
#         mean_inclination = np.mean(incl) * 180. / np.pi
#         err_inclination = np.std(incl) * 180. / np.pi
        
    # return the velocity in deg/hour
    return mean_inclination, err_inclination

def get_semi_major_axis_from_mu(data, distances, single_exposure_time = 30., min_alert_per_exposure = 1):
    """ buggy
    """
    data.sort('dec')

    # Initialise the trajectory
    length = 0.0
    
    # Get unique exposure
    jd_unique = np.unique(data['jd']).data
    n_exposure = len(jd_unique)
    
    # Treat the case where tracklets span several exposures
    mask_exposure = np.ones_like(data['jd'], dtype=bool)
    if np.sum(mask_exposure) < 5:
        return np.nan, np.nan

    if n_exposure > 1:
        # remove exposures with nalert < min_alert_per_exposure
        for k in jd_unique:
            mask_ = data['jd'] == k
            l = np.sum(mask_)
            if l < min_alert_per_exposure:
                mask_exposure[mask_] = False
        
        # Discard if there are not enough alerts left (we want a total of 5 at least)
        if np.sum(mask_exposure) < 5:
            return np.nan, np.nan
        
        # Compute the time lapse between the start of the first and last exposure (0 if the same exposure)
        delta_jd_second = (
            np.max(data['jd'][mask_exposure]) - np.min(data['jd'][mask_exposure])
        ) * 24. * 3600.

        # add the last exposure time
        single_exposure_time = delta_jd_second + 30.
    
    coords = SkyCoord(
        data[mask_exposure]['ra']*u.deg, 
        data[mask_exposure]['dec']*u.deg, 
        distance=distances[mask_exposure]*u.meter, # meters
    ).galactic.cartesian
    xyz = coords.xyz.value.T
    dist_total = np.sum([(i - j)**2 for i, j in zip(xyz[-1], xyz[0])])
    vs = []
    for i in range(len(data[mask_exposure]) - 1):
        x1, y1, z1 = xyz[i]
        x2, y2, z2 = xyz[i+1]
        dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        dt = dist / dist_total * single_exposure_time
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        vz = (z2 - z1) / dt
        vs.append([vx, vy, vz])

    try:
        r = np.array([np.linalg.norm(i) for i in xyz[0:-1]])
        v = np.array([np.linalg.norm(i) for i in vs])
        mu = G.value*M_earth.value
        E = 0.5*(v**2) - mu/r
        a = -mu/(2*E)
    except ValueError:
        print(np.shape(xyz[0:-1]), xyz[0:-1])
        print(np.shape(vs), vs)

    return a # km

def get_class_from_period(period):
    """ Rough classification based on the rotation period.
    
    LEO: P < 2.2 hours
    MEO: 2.2 <= P < 23 hours
    GEO: 23 <= P < 25 hours
    Unknown: P >= 25 hours
    
    Taken from 2012.12549
    
    Parameters
    ----------
    period: float
        Rotation period
    
    Returns
    ----------
    class: str
        Class among: LEO, MEO, GEO, or Unknown
    """
    if period < 2.2:
        return 'LEO'
    elif (period >= 2.2) and (period < 23):
        return 'MEO'
    elif (period >= 23) and (period < 25):
        return 'GEO'
    else:
        return 'Unknown'