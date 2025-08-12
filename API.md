# KineticsTookit (KTK) API Reference\n\nKinetics Toolkit

================

To get started, please consult Kinetics Toolkit's
[website](https://kineticstoolkit.uqam.ca)

>>> import kineticstoolkit as ktk\n\n---\n\n## Module: `kineticstoolkit.lab`\n\nKinetics Toolkit - Lab mode
>>> ===========================
>>>
>>

This module loads Kinetics Toolkit in lab mode. The standard way to use this
module is:

    import kineticstoolkit.lab as ktk

To get started, please consult Kinetics Toolkit's
[website](https://felixchenier.uqam.ca/kineticstoolkit)\n\n## Module: `kineticstoolkit.timeseries`\n\nProvide the TimeSeries and TimeSeriesEvent classes.

The classes defined in this module are accessible directly from the top-
level Kinetics Toolkit's namespace (i.e. ktk.TimeSeries,
ktk.TimeSeriesEvent)\n\n### **Classes**\n\n### `TimeSeries(src: 'None | TimeSeries | pd.DataFrame | ArrayLike' = None, *, time: 'ArrayLike' = [], time_info: 'dict[str, Any]' = {'Unit': 's'}, data: 'dict[str, ArrayLike]' = {}, data_info: 'dict[str, dict[str, Any]]' = {}, events: 'list[TimeSeriesEvent]' = [])`\n\nA class that holds time, data series, events and metadata.

Attributes
----------

time : np.ndarray
    Time attribute as 1-dimension np.array.

data : dict[str, np.ndarray]
    Contains the data, where each element contains a np.array
    which first dimension corresponds to time.

time_info : dict[str, Any]
    Contains metadata relative to time. The default is {"Unit": "s"}

data_info : dict[str, dict[str, Any]]
    Contains optional metadata relative to data. For example, the
    data_info attribute could indicate the unit of data["Forces"]::

    data["Forces"] = {"Unit": "N"}

    To facilitate the management of data_info, please use`ktk.TimeSeries.add_data_info` and `ktk.TimeSeries.remove_data_info`.

events : list[TimeSeriesEvent]
    List of events.

Examples
--------

A TimeSeries can be constructed from another TimeSeries, a Pandas DataFrame
or any array with at least one dimension.

1. Creating an empty TimeSeries:

>>> ktk.TimeSeries()
>>> TimeSeries with attributes:
>>> time: array([], dtype=float64)
>>> data: {}
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

2. Creating a TimeSeries and setting time and data:

>>> ktk.TimeSeries(time=np.arange(0, 10), data={"test":np.arange(0, 10)})
>>> TimeSeries with attributes:
>>> time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

3. Creating a TimeSeries as a copy of another TimeSeries:

>>> ts1 = ktk.TimeSeries(time=np.arange(0, 10), data={"test":np.arange(0, 10)})
>>> ts2 = ktk.TimeSeries(ts1)
>>> ts2
>>> TimeSeries with attributes:
>>> time: array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> data: {'test': array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])}
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

See Also: TimeSeries.copy

4. Creating a TimeSeries from a Pandas DataFrame:

>>> df = pd.DataFrame()
>>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
>>> df["x"] = [0., 1., 2., 3., 4.]
>>> df["y"] = [5., 6., 7., 8., 9.]
>>> df["z"] = [0., 0., 0., 0., 0.]
>>> df
>>> x    y    z
>>> 0.0  0.0  5.0  0.0
>>> 0.1  1.0  6.0  0.0
>>> 0.2  2.0  7.0  0.0
>>> 0.3  3.0  8.0  0.0
>>> 0.4  4.0  9.0  0.0
>>>
>>

>>> ts = ktk.TimeSeries(df)
>>> ts
>>> TimeSeries with attributes:
>>> time: array([0. , 0.1, 0.2, 0.3, 0.4])
>>> data: <dict with 3 entries>
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

>>> ts.data
>>> {'x': array([0., 1., 2., 3., 4.]), 'y': array([5., 6., 7., 8., 9.]), 'z': array([0., 0., 0., 0., 0.])}
>>>
>>

See Also: TimeSeries.from_dataframe

5. Creating a multidimensional TimeSeries from a Pandas DataFrame (using
   brackets in column names):

>>> df = pd.DataFrame()
>>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
>>> df["point[0]"] = [0., 1., 2., 3., 4.]
>>> df["point[1]"] = [5., 6., 7., 8., 9.]
>>> df["point[2]"] = [0., 0., 0., 0., 0.]
>>> df
>>> point[0]  point[1]  point[2]
>>> 0.0       0.0       5.0       0.0
>>> 0.1       1.0       6.0       0.0
>>> 0.2       2.0       7.0       0.0
>>> 0.3       3.0       8.0       0.0
>>> 0.4       4.0       9.0       0.0
>>>
>>

>>> ts = ktk.TimeSeries(df)
>>> ts.data
>>> {'point': array([[0., 5., 0.],
>>> [1., 6., 0.],
>>> [2., 7., 0.],
>>> [3., 8., 0.],
>>> [4., 9., 0.]])}
>>>
>>

See Also: TimeSeries.from_dataframe

6. Creating a multidimensional TimeSeries of higher order from a Pandas
   DataFrame (using brackets and commas in column names):

>>> df = pd.DataFrame()
>>> df.index = [0., 0.1, 0.2, 0.3, 0.4]  # Time in seconds
>>> df["rot[0,0]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
>>> df["rot[0,1]"] = -np.sin([0., 0.1, 0.2, 0.3, 0.4])
>>> df["rot[1,0]"] = np.sin([0., 0.1, 0.2, 0.3, 0.4])
>>> df["rot[1,1]"] = np.cos([0., 0.1, 0.2, 0.3, 0.4])
>>> df["trans[0]"] = [0., 0.1, 0.2, 0.3, 0.4]
>>> df["trans[1]"] = [5., 6., 7., 8., 9.]
>>> df
>>> rot[0,0]  rot[0,1]  rot[1,0]  rot[1,1]  trans[0]  trans[1]
>>> 0.0  1.000000 -0.000000  0.000000  1.000000       0.0       5.0
>>> 0.1  0.995004 -0.099833  0.099833  0.995004       0.1       6.0
>>> 0.2  0.980067 -0.198669  0.198669  0.980067       0.2       7.0
>>> 0.3  0.955336 -0.295520  0.295520  0.955336       0.3       8.0
>>> 0.4  0.921061 -0.389418  0.389418  0.921061       0.4       9.0
>>>
>>

>>> ts = ktk.TimeSeries(df)
>>> ts.data
>>> {'rot': array([[[ 1.        , -0.        ],
>>> [ 0.        ,  1.        ]],
>>> `<BLANKLINE>`
>>> [[ 0.99500417, -0.09983342],
>>> [ 0.09983342,  0.99500417]],
>>> `<BLANKLINE>`
>>> [[ 0.98006658, -0.19866933],
>>> [ 0.19866933,  0.98006658]],
>>> `<BLANKLINE>`
>>> [[ 0.95533649, -0.29552021],
>>> [ 0.29552021,  0.95533649]],
>>> `<BLANKLINE>`
>>> [[ 0.92106099, -0.38941834],
>>> [ 0.38941834,  0.92106099]]]), 'trans': array([[0. , 5. ],
>>> [0.1, 6. ],
>>> [0.2, 7. ],
>>> [0.3, 8. ],
>>> [0.4, 9. ]])}
>>>
>>

See Also: TimeSeries.from_dataframe

7. Creating a TimeSeries from any array (results in a TimeSeries with a
   single data key named "data" and with a matching time property with a
   period of 1 second - unless time attribute is also defined):

>>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5])
>>> TimeSeries with attributes:
>>> time: array([0., 1., 2., 3., 4.])
>>> data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

>>> ktk.TimeSeries([0.1, 0.2, 0.3, 0.4, 0.5], time=[0.1, 0.2, 0.3, 0.4, 0.5])
>>> TimeSeries with attributes:
>>> time: array([0.1, 0.2, 0.3, 0.4, 0.5])
>>> data: {'data': array([0.1, 0.2, 0.3, 0.4, 0.5])}
>>> time_info: {'Unit': 's'}
>>> data_info: {}
>>> events: []
>>>
>>

See Also: TimeSeries.from_array\n\n---\n\n### `TimeSeriesDataDict(source: 'dict' = {})`\n\nData dictionary that checks sizes and converts to NumPy arrays.\n\n---\n\n### `TimeSeriesEvent(time: 'float' = 0.0, name: 'str' = 'event') -> None`\n\nDefine an event in a TimeSeries.

This class is rarely used by itself, it is easier to use `TimeSeries`'
methods to manage events.

Attributes
----------

time : float
    Event time.

name : str
    Event name. Does not need to be unique.

Example
-------

>>> event = ktk.TimeSeriesEvent(time=1.5, name="event_name")
>>> event
>>> TimeSeriesEvent(time=1.5, name='event_name')\n\n---\n\n### `TimeSeriesEventList(source: 'list' = [])`\n\nEvent list that ensures every element is a TimeSeriesEvent.\n\n---\n\n## Module: `kineticstoolkit.filters`\n\nProvide standard filters for TimeSeries.\n\n### **Functions**\n\n### `butter(ts: kineticstoolkit.timeseries.TimeSeries, /, fc: float | tuple[float, float], *, order: int = 2, btype: str = 'lowpass', filtfilt: bool = True) -> kineticstoolkit.timeseries.TimeSeries`\n\nApply a Butterworth filter to a TimeSeries.
>>>
>>

Filtering occurs on the first axis (time). If the TimeSeries contains
missing samples, a warning is issued, missing samples are interpolated
using a first-order interpolation before filtering, and then replaced by
np.nan in the filtered signal.

Parameters
----------

ts
    Input TimeSeries.
fc
    Cut-off frequency in Hz. This is a float for single-frequency filters
    (lowpass, highpass), or a tuple of two floats (e.g., (10., 13.)
    for two-frequency filters (bandpass, bandstop)).
order
    Optional. Order of the filter. Default is 2.
btype
    Optional. Can be either "lowpass", "highpass", "bandpass" or
    "bandstop". Default is "lowpass".
filtfilt
    Optional. If True, the filter is applied two times in reverse direction
    to eliminate time lag. If False, the filter is applied only in forward
    direction. Default is True.

Returns
-------

TimeSeries
    A copy of the input TimeSeries, which each data being filtered.

Raises
------

ValueError
    If sample rate is not constant, or if there is no data to
    filter.\n\n---\n\n### `deriv(ts: kineticstoolkit.timeseries.TimeSeries, /, n: int = 1) -> kineticstoolkit.timeseries.TimeSeries`\n\nCalculate the nth numerical derivative.

Filtering occurs on the first axis (time). The sample rate must be
constant.

Parameters
----------

ts
    Input timeseries

n
    Order of the derivative.

Returns
-------

TimeSeries
    A copy of the input TimeSeries, which each data being derived. The
    length of the resulting TimeSeries is one less than `ts`.

Raises
------

ValueError
    If sample rate is not constant, or if there is no data to
    filter.

Example
-------

>>> ts = ktk.TimeSeries(time=np.arange(0, 0.5, 0.1))
>>> ts = ts.add_data("test", np.array([0.0, 0.0, 1.0, 1.0, 0.0]))
>>>
>>

>>> # Source data
>>>
>>> ts.time
>>> array([0. , 0.1, 0.2, 0.3, 0.4])
>>> ts.data["test"]
>>> array([0., 0., 1., 1., 0.])
>>>
>>

>>> # First derivative
>>>
>>> ts1 = ktk.filters.deriv(ts)
>>>
>>

>>> ts1.time
>>> array([0.05, 0.15, 0.25, 0.35])
>>> ts1.data["test"]
>>> array([  0.,  10.,   0., -10.])
>>>
>>

>>> # Second derivative
>>>
>>> ts2 = ktk.filters.deriv(ts, n=2)
>>>
>>

>>> ts2.time
>>> array([0.1, 0.2, 0.3])
>>> ts2.data["test"]
>>> array([ 100., -100., -100.])\n\n---\n\n### `median(ts: kineticstoolkit.timeseries.TimeSeries, /, window_length: int = 3) -> kineticstoolkit.timeseries.TimeSeries`\n\nCalculate a moving median.
>>>
>>

Filtering occurs on the first axis (time).

Parameters
----------

ts
    Input TimeSeries

window_length
    Optional. Kernel size, must be odd. The default is 3.

Example
-------

>>> ts = ktk.TimeSeries(time=np.arange(0, 6))
>>> ts = ts.add_data("test", [10., 11., 11., 20., 14., 15.])
>>> ts2 = ktk.filters.median(ts)
>>> ts2.data["test"]
>>> array([10., 11., 11., 14., 15., 15.])\n\n---\n\n### `savgol(ts: kineticstoolkit.timeseries.TimeSeries, /, *, window_length: int, poly_order: int, deriv: int = 0) -> kineticstoolkit.timeseries.TimeSeries`\n\nApply a Savitzky-Golay filter on a TimeSeries.
>>>
>>

Filtering occurs on the first axis (time). If the TimeSeries contains
missing samples, a warning is issued, missing samples are interpolated
using a first-order interpolation before filtering, and then replaced by
np.nan in the filtered signal.

Parameters
----------

ts
    Input TimeSeries
window_length
    The length of the filter window. window_length must be a positive
    odd integer less or equal than the length of the TimeSeries.
poly_order
    The order of the polynomial used to fit the samples. polyorder must be
    less than window_length.
deriv
    Optional. The order of the derivative to compute. The default is 0,
    which means to filter the data without differentiating.

Returns
-------

TimeSeries
    A copy of the input TimeSeries, which each data being filtered.

Raises
------

ValueError
    If sample rate is not constant, or if there is no data to
    filter.

See Also
--------

ktk.filters.smooth\n\n---\n\n### `smooth(ts: kineticstoolkit.timeseries.TimeSeries, /, window_length: int) -> kineticstoolkit.timeseries.TimeSeries`\n\nApply a smoothing (moving average) filter on a TimeSeries.

Filtering occurs on the first axis (time). If the TimeSeries contains
missing samples, a warning is issued, missing samples are interpolated
using a first-order interpolation before filtering, and then replaced by
np.nan in the filtered signal.

Parameters
----------

ts
    Input TimeSeries.
window_length
    The length of the filter window. window_length must be a positive
    odd integer less or equal than the length of the TimeSeries.

Returns
-------

TimeSeries
    A copy of the input TimeSeries, which each data being filtered.

Raises
------

ValueError
    If sample rate is not constant, or if there is no data to
    filter.

See Also
--------

ktk.filters.savgol\n\n---\n\n## Module: `kineticstoolkit.kinematics`\n\nProvide functions related to kinematics analysis.\n\n### **Functions**\n\n### `create_cluster(markers: kineticstoolkit.timeseries.TimeSeries, /, names: list[str]) -> dict[str, numpy.ndarray]`\n\nCreate a cluster definition based on a static acquisition.

Parameters
----------

markers
    Marker trajectories during a static acquisition.
names
    The markers that define the cluster.

Returns
-------

dict
    Dictionary where each entry represents the local position of a marker
    in an arbitrary coordinate system.

Note
----

0.10.0: Parameter `marker_names` was changed to `names`

See Also
--------

ktk.kinematics.extend_cluster
ktk.kinematics.track_cluster\n\n---\n\n### `extend_cluster(markers: kineticstoolkit.timeseries.TimeSeries, /, cluster: dict[str, numpy.ndarray], name: str) -> dict[str, numpy.ndarray]`\n\nAdd a point to an existing cluster.

Parameters
----------

markers
    TimeSeries that includes the new point trajectory, along with point
    trajectories from the cluster definition.
cluster
    The source cluster to add a new point to.
name
    The name of the point to add (data key of the markers TimeSeries).

Returns
-------

dict[str, np.ndarray]
    A copy of the initial cluster, with the added point.

Note
----

0.10.0: Parameter `new_point` was changed to `name`

See Also
--------

ktk.kinematics.create_cluster
ktk.kinematics.track_cluster\n\n---\n\n### `track_cluster(markers: kineticstoolkit.timeseries.TimeSeries, /, cluster: dict[str, numpy.ndarray], *, include_lcs: bool = False, lcs_name: str = 'LCS') -> kineticstoolkit.timeseries.TimeSeries`\n\nFit a cluster to a TimeSeries of point trajectories.

This function fits a cluster to a TimeSeries and reconstructs a solidified
version of all the points defined in this cluster.

Parameters
----------

markers
    A TimeSeries that contains point trajectories as Nx4 arrays.
cluster
    A cluster definition as returned by ktk.kinematics.create_cluster().
include_lcs
    Optional. If True, return an additional entry in the output
    TimeSeries, that is the Nx4x4 frame series corresponding to the
    tracked cluster's local coordinate system. The default is False.
lcs_name
    Optional. Name of the TimeSeries data entry for the tracked local
    coordinate system. The default is "LCS".

Returns
-------

TimeSeries
    A TimeSeries with the trajectories of all cluster points.

See Also
--------

ktk.kinematics.create_cluster
ktk.kinematics.track_cluster\n\n---\n\n## Module: `kineticstoolkit.geometry`\n\nProvide 3D geometry and linear algebra functions related to biomechanics.

Note
----

As a convention, the first dimension of every array is always N and corresponds
to time.\n\n### **Functions**\n\n### `create_point_series(array: Optional[kineticstoolkit.typing_.ArrayLike] = None, *, x: Optional[kineticstoolkit.typing_.ArrayLike] = None, y: Optional[kineticstoolkit.typing_.ArrayLike] = None, z: Optional[kineticstoolkit.typing_.ArrayLike] = None, length: int | None = None) -> numpy.ndarray`\n\nCreate an Nx4 point series ([[x, y, z, 1.0], ...]).

**Single array**

To create a point series based on a single array, use this form::

    create_point_series(
        array: ArrayLike | None = None,
        *,
        length: int | None = None,
    ) -> np.ndarray:

**Multiple arrays**

To create a point series based on multiple arrays (e.g., x, y, z), use
this form::

    create_point_series(
        *,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        length: int | None = None,
    ) -> np.ndarray:

Parameters
----------

array
    Used in single array input form.
    Array of one of these shapes where N corresponds to time:
    (N,), (N, 1): forms a point series on the x axis, with y=0 and z=0.
    (N, 2): forms a point series on the x, y axes, with z=0.
    (N, 3), (N, 4): forms a point series on the x, y, z axes.

x
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the x values. If not
    provided, x values are filled with zero.

y
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the y values. If not
    provided, y values are filled with zero.

z
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the z values. If not
    provided, z values are filled with zero.

length
    The number of samples in the resulting point series. If there is only
    one sample in the original array, this one sample will be duplicated
    to match length. Otherwise, an error is raised if the input
    array does not match length.

Returns
-------

array
    An Nx4 array with every sample being [x, y, z, 1.0].

Raises
------

ValueError
    If the inputs have incorrect dimensions.

Examples
--------

Single input form::

    # A series of 2 samples with x, y defined
    >>> ktk.geometry.create_point_series([[1.0, 2.0], [4.0, 5.0]])
    array([[1., 2., 0., 1.],
           [4., 5., 0., 1.]])

    # A series of 2 samples with x, y, z defined
    >>> ktk.geometry.create_point_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    array([[1., 2., 3., 1.],
           [4., 5., 6., 1.]])

    # Samething
    >>> ktk.geometry.create_point_series([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]])
    array([[1., 2., 3., 1.],
           [4., 5., 6., 1.]])

Multiple inputs form::

    # A series of 2 samples with x, z defined
    >>> ktk.geometry.create_point_series(x=[1.0, 2.0, 3.0], z=[4.0, 5.0, 6.0])
    array([[1., 0., 4., 1.],
           [2., 0., 5., 1.],
           [3., 0., 6., 1.]])\n\n---\n\n###`create_transform_series(matrices: Optional[kineticstoolkit.typing_.ArrayLike] = None, *, angles: Optional[kineticstoolkit.typing_.ArrayLike] = None, seq: str | None = None, degrees: bool = False, quaternions: Optional[kineticstoolkit.typing_.ArrayLike] = None, scalar_first: bool = False, x: Optional[kineticstoolkit.typing_.ArrayLike] = None, y: Optional[kineticstoolkit.typing_.ArrayLike] = None, z: Optional[kineticstoolkit.typing_.ArrayLike] = None, xy: Optional[kineticstoolkit.typing_.ArrayLike] = None, xz: Optional[kineticstoolkit.typing_.ArrayLike] = None, yz: Optional[kineticstoolkit.typing_.ArrayLike] = None, positions: Optional[kineticstoolkit.typing_.ArrayLike] = None, length: int | None = None) -> numpy.ndarray`\n\nCreate an Nx4x4 transform series from multiple input forms.

**Matrix input form**

If the input is a series of 3x3 rotation matrices or 4x4 homogeneous
transforms, use this form::

    ktk.geometry.to_transform_series(
        matrices: ArrayLike,
        *,
        positions: ArrayLike | None = None,
        length: int | None = None,
        ) -> np.ndarray

**Angle input form**

If the input is a series of Euler/cardan angles, use this form::

    ktk.geometry.to_transform_series(
        *,
        angles: ArrayLike,
        seq: str,
        degrees: bool = False,
        positions: ArrayLike | None = None,
        length: int | None = None,
        ) -> np.ndarray

**Quaternion input form**

If the input is a series of quaternions, use this form::

    ktk.geometry.to_transform_series(
        *,
        quaternions: ArrayLike,
        scalar_first: bool = False,
        positions: ArrayLike | None = None,
        length: int | None = None,
        ) -> np.ndarray

**Vector input form (using cross-product)**

To create transform series that represent a local coordinate system based
on the cross product of different vectors, use this form, where one of
{x, y, z} and one of {xy, xz, yz} must be defined::

    ktk.geometry.to_transform_series(
        *,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        xy: ArrayLike | None = None,
        xz: ArrayLike | None = None,
        yz: ArrayLike | None = None,
        positions: ArrayLike | None = None,
        length: int | None = None,
        ) -> np.ndarray

With this input form, x, y or z sets the first axis of the local coordinate
system. Then, xy, xz or yz forms a plane with the first vector; the second
axis is the cross product of both vectors (perpendicular to this plane).
Finally, the third axis is the cross product of the two first axes.

Parameters
----------

matrices
    Used in the matrix input form.
    Nx3x3 series or rotations or Nx4x4 series of homogeneous transforms.

angles
    Used in the angles input form.
    Series of angles, either of shape (N,) or (N, 1) for rotations around
    only one axis, or (N, 2) or (N, 3) for rotations around consecutive
    axes.

seq
    Used in the angles input form.
    Specifies the sequence of axes for successive rotations. Up to 3
    characters belonging to {"X", "Y", "Z"} for intrinsic rotations
    (moving axes), or {"x", "y", "z"} for extrinsic rotations (fixed
    axes). Extrinsic and intrinsic rotations cannot be mixed in one
    function call.

degrees
    Used in the angles input form.
    Optional. If True, then the given angles are in degrees, otherwise
    they are in radians. Default is False (radians).

quaternions
    Used in the quaternions input form. Nx4 series of quaternions.

scalar_first
    Used in the quaternions input form.
    Optional. If True, the quaternion order is (w, x, y, z). If False,
    the quaternion order is (x, y, z, w). Default is False.

x, y, z
    Used in the vector input form.
    Define either `x`, `y` or `z`. A series of N vectors (Nx4) that
    define the {x|y|z} axis of the frames to be created.

xy
    Used in the vector input form.
    Only if `x` or `y` is specified. A series of N vectors (Nx4) in the xy
    plane, to create `z` using (x cross xy) or (xy cross y). Choose vectors
    that point roughly in the +x or +y direction.

xz
    Used in the vector input form.
    Only if `x` or `z` is specified. A series of N vectors (Nx4) in the xz
    plane, to create `y` using (xz cross x) or (z cross xz). Choose vectors
    that point roughly in the +x or +z direction.

yz
    Used in the vector input form.
    Only if `y` or `z` is specified. A series of N vectors (Nx4) in the yz
    plane, to create `x` using (y cross yz) or (yz cross z). Choose vectors
    that point roughly in the +y or +z direction.

positions
    Optional. An Nx2, Nx3 or Nx4 point series that defines the position
    component (fourth column) of the transforms. Default value is
    [[0.0, 0.0, 0.0, 1.0]]. If the input is an Nx4x4 frame series and
    therefore already has positions, then the existing positions are kept
    unless `positions` is specified.

length
    Optional. The number of samples in the resulting series. If there
    is only one sample in the original array, this one sample will be
    duplicated to match length. Otherwise, an error is raised if the input
    array does not match length.

Returns
-------

np.ndarray
    An Nx4x4 transform series.

Examples
--------

**Matrix input**

Convert a 2x3x3 rotation matrix series and a 1x4 position series to
an 2x4x4 homogeneous transform series:

>>> positions = [[0.5, 0.6, 0.7]]
>>> rotations = [[[ 1.,  0.,  0.],
>>> ...              [ 0.,  1.,  0.],
>>> ...              [ 0.,  0.,  1.]],
>>> ...             [[ 1.,  0.,  0.],
>>> ...              [ 0.,  0., -1.],
>>> ...              [ 0.,  1.,  0.]]]
>>> ktk.geometry.create_transform_series(rotations, positions=positions)
>>> array([[[ 1. ,  0. ,  0. ,  0.5],
>>> [ 0. ,  1. ,  0. ,  0.6],
>>> [ 0. ,  0. ,  1. ,  0.7],
>>> [ 0. ,  0. ,  0. ,  1. ]],
>>> `<BLANKLINE>`
>>> [[ 1. ,  0. ,  0. ,  0.5],
>>> [ 0. ,  0. , -1. ,  0.6],
>>> [ 0. ,  1. ,  0. ,  0.7],
>>> [ 0. ,  0. ,  0. ,  1. ]]])
>>>
>>

**Angle input**

Create a series of two homogeneous transforms that rotates 0, then 90
degrees around x:

>>> ktk.geometry.create_transform_series(angles=[0, 90], seq="x", degrees=True)
>>> array([[[ 1.,  0.,  0.,  0.],
>>> [ 0.,  1.,  0.,  0.],
>>> [ 0.,  0.,  1.,  0.],
>>> [ 0.,  0.,  0.,  1.]],
>>> `<BLANKLINE>`
>>> [[ 1.,  0.,  0.,  0.],
>>> [ 0.,  0., -1.,  0.],
>>> [ 0.,  1.,  0.,  0.],
>>> [ 0.,  0.,  0.,  1.]]])\n\n---\n\n### `create_vector_series(array: Optional[kineticstoolkit.typing_.ArrayLike] = None, *, x: Optional[kineticstoolkit.typing_.ArrayLike] = None, y: Optional[kineticstoolkit.typing_.ArrayLike] = None, z: Optional[kineticstoolkit.typing_.ArrayLike] = None, length: int | None = None) -> numpy.ndarray`\n\nCreate an Nx4 vector series ([[x, y, z, 0.0], ...]).
>>>
>>

**Single array**

To create a vector series based on a single array, use this form::

    create_vector_series(
        array: ArrayLike | None = None,
        *,
        length: int | None = None,
    ) -> np.ndarray:

**Multiple arrays**

To create a vector series based on multiple arrays (e.g., x, y, z), use
this form::

    create_vector_series(
        *,
        x: ArrayLike | None = None,
        y: ArrayLike | None = None,
        z: ArrayLike | None = None,
        length: int | None = None,
    ) -> np.ndarray:

Parameters
----------

array
    Used in single array input form.
    Array of one of these shapes where N corresponds to time:
    (N,), (N, 1): forms a vector series on the x axis, with y=0 and z=0.
    (N, 2): forms a vector series on the x, y axes, with z=0.
    (N, 3), (N, 4): forms a vector series on the x, y, z axes.

x
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the x values. If not
    provided, x values are filled with zero.

y
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the y values. If not
    provided, y values are filled with zero.

z
    Used in multiple arrays input form.
    Optional. Array of shape (N,) that contains the z values. If not
    provided, z values are filled with zero.

length
    The number of samples in the resulting vector series. If there is only
    one sample in the original array, this one sample will be duplicated
    to match length. Otherwise, an error is raised if the input
    array does not match length.

Returns
-------

array
    An Nx4 array with every sample being [x, y, z, 0.0].

Raises
------

ValueError
    If the inputs have incorrect dimensions.

Examples
--------

Single input form::

    # A series of 2 samples with x, y defined
    >>> ktk.geometry.create_vector_series([[1.0, 2.0], [4.0, 5.0]])
    array([[1., 2., 0., 0.],
           [4., 5., 0., 0.]])

    # A series of 2 samples with x, y, z defined
    >>> ktk.geometry.create_vector_series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    array([[1., 2., 3., 0.],
           [4., 5., 6., 0.]])

    # Samething
    >>> ktk.geometry.create_vector_series([[1.0, 2.0, 3.0, 1.0], [4.0, 5.0, 6.0, 1.0]])
    array([[1., 2., 3., 0.],
           [4., 5., 6., 0.]])

Multiple inputs form::

    # A series of 2 samples with x, z defined
    >>> ktk.geometry.create_vector_series(x=[1.0, 2.0, 3.0], z=[4.0, 5.0, 6.0])
    array([[1., 0., 4., 0.],
           [2., 0., 5., 0.],
           [3., 0., 6., 0.]])\n\n---\n\n###`get_angles(T: kineticstoolkit.typing_.ArrayLike, seq: str, degrees: bool = False, flip: bool = False) -> numpy.ndarray`\n\nExtract Euler angles from a transform series.

In case of gimbal lock, a warning is raised, and the third angle is set to
zero. Note however that the returned angles still represent the correct
rotation.

Parameters
----------

T
    An Nx4x4 transform series.

seq
    Specifies the sequence of axes for successive rotations. Up to 3
    characters belonging to the set {"X", "Y", "Z"} for intrinsic
    rotations (moving axes), or {"x", "y", "z"} for extrinsic rotations
    (fixed axes). Adjacent axes cannot be the same. Extrinsic and
    intrinsic rotations cannot be mixed in one function call.

degrees
    If True, the returned angles are in degrees. If False, they are in
    radians. Default is False.

flip
    Return an alternate sequence with the second angle inverted, but that
    leads to the same rotation matrices. See below for more information.

Returns
-------

np.ndarray
    An Nx3 series of Euler angles, with the second dimension containing
    the first, second and third angles, respectively.

Notes
-----

The range of the returned angles is dependent on the `flip` parameter. If
`flip` is False:

- First angle belongs to [-180, 180] degrees (both inclusive)
- Second angle belongs to:

  - [-90, 90] degrees if all axes are different. e.g., xyz
  - [0, 180] degrees if first and third axes are the same e.g., zxz
- Third angle belongs to [-180, 180] degrees (both inclusive)

If `flip` is True:

- First angle belongs to [-180, 180] degrees (both inclusive)
- Second angle belongs to:

  - [-180, -90], [90, 180] degrees if all axes are different. e.g., xyz
  - [-180, 0] degrees if first and third axes are the same e.g., zxz
- Third angle belongs to [-180, 180] degrees (both inclusive)

This function is a wrapper for scipy.transform.Rotation.as_euler. Please
consult scipy help for more help on intrinsic/extrinsic angles and the
`seq` parameter.\n\n---\n\n### `get_global_coordinates(local_coordinates: kineticstoolkit.typing_.ArrayLike, reference_frames: kineticstoolkit.typing_.ArrayLike) -> numpy.ndarray`\n\nExpress local coordinates in the global reference frame.

Parameters
----------

local_coordinates
    The local coordinates, as a series of N points, vectors or matrices.
    For example:

    - A series of N points or vectors : Nx4
    - A series of N set of M points or vectors : Nx4xM
    - A series of N 4x4 transformation matrices : Nx4x4

reference_frames
    An Nx4x4 transform series that represents the local coordinate system.

Returns
-------

np.ndarray
    Series of global coordinates in the same shape as
    `local_coordinates`.

See Also
--------

ktk.geometry.get_local_coordinates\n\n---\n\n### `get_local_coordinates(global_coordinates: kineticstoolkit.typing_.ArrayLike, reference_frames: kineticstoolkit.typing_.ArrayLike) -> numpy.ndarray`\n\nExpress global coordinates in local reference frames.

Parameters
----------

global_coordinates
    The global coordinates, as a series of N points, vectors or matrices.
    For example:

    - A series of N points or vectors : Nx4
    - A series of N set of M points or vectors : Nx4xM
    - A series of N 4x4 transformation matrices : Nx4x4

reference_frames
    An Nx4x4 transform series that represents the local coordinate system.

Returns
-------

np.ndarray
    Series of local coordinates in the same shape as
    `global_coordinates`.

See Also
--------

ktk.geometry.get_global_coordinates\n\n---\n\n### `get_quaternions(T: kineticstoolkit.typing_.ArrayLike, canonical: bool = False, scalar_first: bool = False) -> numpy.ndarray`\n\nExtract quaternions from a transform series.

Parameters
----------

T
    An Nx4x4 transform series.

canonical
    Whether to map the redundant double cover of rotation space to a
    unique "canonical" single cover. If True, then the quaternion is
    chosen from {q, -q} such that the w term is positive. If the w term is
    0, then the quaternion is chosen such that the first nonzero term of
    the x, y, and z terms is positive. Default is False.

scalar_first
    Optional. If True, the quaternion order is (w, x, y, z). If False,
    the quaternion order is (x, y, z, w). Default is False.

Returns
-------

np.ndarray
    An Nx4 series of quaternions.\n\n---\n\n### `is_point_series(array: kineticstoolkit.typing_.ArrayLike) -> bool`\n\nCheck that the input is an Nx4 point series ([[x, y, z, 1.0], ...]).

Parameters
----------

array
    Array where the first dimension corresponds to time.

Returns
-------

bool
    True if every sample (other than NaNs) of the input array is a point
    (an array of length 4 with the last component being 1.0)\n\n---\n\n### `is_transform_series(array: kineticstoolkit.typing_.ArrayLike, /) -> bool`\n\nCheck that the input is an Nx4x4 series of homogeneous transforms.

Parameters
----------

array
    Array where the first dimension corresponds to time.

Returns
-------

bool
    True if every sample (other than NaNs) of the input array is a
    4x4 homogeneous transform.\n\n---\n\n### `is_vector_series(array: kineticstoolkit.typing_.ArrayLike) -> bool`\n\nCheck that the input is an Nx4 vector series ([[x, y, z, 0.0], ...]).

Parameters
----------

array
    Array where the first dimension corresponds to time.

Returns
-------

bool
    True if every sample (other than NaNs) of the input array is a vector
    (an array of length 4 with the last component being 0.0)\n\n---\n\n### `isnan(array: kineticstoolkit.typing_.ArrayLike, /) -> numpy.ndarray`\n\nCheck which samples contain at least one NaN.

Parameters
----------

array
    Array where the first dimension corresponds to time.

Returns
-------

np.ndarray
    Array of bool that is the same size of input's first dimension, with
    True for the samples that contain at least one NaN.\n\n---\n\n### `matmul(op1: kineticstoolkit.typing_.ArrayLike, op2: kineticstoolkit.typing_.ArrayLike, /) -> numpy.ndarray`\n\nMatrix multiplication between series of matrices.

This function is a wrapper for numpy's matmul function (operator @), that
uses Kinetics Toolkit's convention that the first dimension always
corresponds to time, to broadcast time correctly between operands.

Parameters
----------

op1
    Series of floats, vectors or matrices.
op2
    Series of floats, vectors or matrices.

Returns
-------

np.ndarray
    The product, usually as a series of Nx4 or Nx4xM matrices.

Example
-------

A matrix multiplication between one matrix and a series of 3 vectors
results in a series of 3 vectors.

>>> import kineticstoolkit as ktk
>>> mat_series = np.array([[[2.0, 0.0], [0.0, 1.0]]])
>>> vec_series = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
>>> ktk.geometry.matmul(mat_series, vec_series)
>>> array([[ 8.,  5.],
>>> [12.,  7.],
>>> [16.,  9.]])\n\n---\n\n### `mirror(coordinates, /, axis: str = 'z')`\n\nMirror a series of coordinates.
>>>
>>

Parameters
----------

coordinates
    ArrayLike of shape (N, ...): the coordinates to mirror.

axis
    Can be either "x", "y" or "z". The axis to mirror through. The default
    is "z".

Returns
-------

np.ndarray
    ArrayLike of shape (N, ...): the mirrored coordinates.

See Also
--------

ktk.geometry.rotate, ktk.geometry.translate, ktk.geometry.scale

Examples
--------

Mirror the point (1, 2, 3) along the x, y and z axes respectively:

    >>> import kineticstoolkit.lab as ktk
    >>> import numpy as np
    >>> p = np.array([[1.0, 2.0, 3.0, 1.0]])

    >>> ktk.geometry.mirror(p, "x")
    array([[-1., 2., 3., 1.]])

    >>> ktk.geometry.mirror(p, "y")
    array([[ 1., -2., 3., 1.]])

    >>> ktk.geometry.mirror(p, "z")
    array([[ 1., 2., -3., 1.]])\n\n---\n\n###`register_points(global_points: kineticstoolkit.typing_.ArrayLike, local_points: kineticstoolkit.typing_.ArrayLike) -> numpy.ndarray`\n\nFind the homogeneous transforms between two series of point clouds.

Parameters
----------

global_points
    Destination points as an Nx4xM series of N sets of M points.
local_points
    Local points as an array of shape Nx4xM series of N sets of M points.
    global_points and local_points must have the same shape.

Returns
-------

np.ndarray
    Array of shape Nx4x4, expressing a series of 4x4 homogeneous
    transforms.\n\n---\n\n### `rotate(coordinates, /, seq: str, angles: kineticstoolkit.typing_.ArrayLike, *, degrees: bool = False) -> numpy.ndarray`\n\nRotate a series of coordinates along given axes.

Parameters
----------

coordinates
    ArrayLike of shape (N, ...): the coordinates to rotate.

seq
    Specifies sequence of axes for rotations. Up to 3 characters
    belonging to the set {"X", "Y", "Z"} for intrinsic rotations (moving
    axes), or {"x", "y", "z"} for extrinsic rotations (fixed axes).
    Extrinsic and intrinsic rotations cannot be mixed in one function call.

angles
    ArrayLike of shape (N,) or (N, [1 or 2 or 3]). Angles are
    specified in radians (if degrees is False) or degrees (if degrees is
    True).

    For a single-character`seq`, `angles` can be:

    - ArrayLike with shape (N,), where each`angle[i]` corresponds to a
      single rotation;
    - ArrayLike with shape (N, 1), where each `angle[i, 0]` corresponds
      to a single rotation.

    For 2- and 3-character`seq`, `angles` is an ArrayLike with shape
    (N, W) where each `angle[i, :]` corresponds to a sequence of Euler
    angles and W is the length of `seq`.

degrees
    If True, then the given angles are in degrees. Default is False.

Returns
-------

np.ndarray
    ArrayLike of shape (N, ...): the rotated coordinates.

See Also
--------

ktk.geometry.translate, ktk.geometry.scale, ktk.geometry.mirror

Examples
--------

Rotate the point (1, 0, 0) by theta degrees around z, then by 45 degrees
around y, for theta in [0, 10, 20, 30, 40]:

    >>> import kineticstoolkit.lab as ktk
    >>> angles = np.array([[0, 45], [10, 45], [20, 45], [30, 45], [40, 45]])
    >>> ktk.geometry.rotate([[1, 0, 0, 1]], "zx", angles, degrees=True)
    array([[1.        , 0.        , 0.        , 1.        ],
           [0.98480775, 0.1227878 , 0.1227878 , 1.        ],
           [0.93969262, 0.24184476, 0.24184476, 1.        ],
           [0.8660254 , 0.35355339, 0.35355339, 1.        ],
           [0.76604444, 0.45451948, 0.45451948, 1.        ]])\n\n---\n\n###`scale(coordinates, /, scales)`\n\nScale a series of coordinates.

Parameters
----------

coordinates
    ArrayLike of shape (N, ...): the coordinates to scale.

scales
    ArrayLike of shape (N, ) that corresponds to the scale to apply
    uniformly on the three axes.

Returns
-------

np.ndarray
    ArrayLike of shape (N, ...): the scaled coordinates.

See Also
--------

ktk.geometry.rotate, ktk.geometry.translate, ktk.geometry.mirror

Examples
--------

Scale the point (1, 0, 0) by x, for x in [0, 1, 2, 3, 4]:

    >>> import kineticstoolkit.lab as ktk
    >>> s = np.array([0, 1, 2, 3, 4])
    >>> ktk.geometry.scale([[1, 0, 0, 1]], s)
    array([[0., 0., 0., 1.],
           [1., 0., 0., 1.],
           [2., 0., 0., 1.],
           [3., 0., 0., 1.],
           [4., 0., 0., 1.]])\n\n---\n\n###`translate(coordinates, /, translations)`\n\nTranslate a series of coordinates.

Parameters
----------

coordinates
    ArrayLike of shape (N, ...): the coordinates to translate.

translations
    ArrayLike of shape (N, 3) or (N, 4): the translation on each axis
    (x, y, z).

Returns
-------

np.ndarray
    ArrayLike of shape (N, ...): the translated coordinates.

See Also
--------

ktk.geometry.rotate, ktk.geometry.scale, ktk.geometry.mirror

Examples
--------

Translate the point (1, 0, 0) by (x, 1, 0), for x in [0, 1, 2, 3, 4]:

    >>> import kineticstoolkit.lab as ktk
    >>> t = np.array([[0, 1, 0], [1, 1, 0], [2, 1, 0], [3, 1, 0], [4, 1, 0]])
    >>> ktk.geometry.translate([[1, 0, 0, 1]], t)
    array([[1., 1., 0., 1.],
           [2., 1., 0., 1.],
           [3., 1., 0., 1.],
           [4., 1., 0., 1.],
           [5., 1., 0., 1.]])\n\n---\n\n## Module:`kineticstoolkit.cycles`\n\nIdentify cycles and time-normalize data.\n\n### **Functions**\n\n### `detect_cycles(ts: kineticstoolkit.timeseries.TimeSeries, data_key: str, *, event_names: tuple[str, str] = ('phase1', 'phase2'), thresholds: tuple[float, float] = (0.0, 1.0), directions: tuple[str, str] = ('rising', 'falling'), min_durations: tuple[float, float] = (0.0, 0.0), max_durations: tuple[float, float] = (inf, inf), min_peak_heights: tuple[float, float] = (-inf, -inf), max_peak_heights: tuple[float, float] = (inf, inf)) -> kineticstoolkit.timeseries.TimeSeries`\n\nDetect cycles in a TimeSeries based on a dual threshold approach.

This function detects biphasic cycles and identifies the transitions as
new events in the output TimeSeries. These new events are named:

- event_names[0]:
  corresponds to the start of phase 1
- event_names[1]:
  corresponds to the start of phase 2
- "_":
  corresponds to the end of the cycle.

Parameters
----------

ts
    TimeSeries to analyze.
data_key
    Name of the data key to analyze in the TimeSeries. This data must be
    unidimensional.
event_names
    Optional. Event names to add in the output TimeSeries. Default is
    ("phase1", "phase2").
thresholds
    Optional. Values to cross to register phase changes. Default is
    [0., 1.].
directions
    Optional. Directions to cross thresholds to register phase changes.
    Either ("rising", "falling") or ("falling", "rising"). Default is
    ("rising", "falling").
min_durations
    Optional. Minimal phase durations in seconds. Default is (0.0, 0.0).
max_durations
    Optional. Maximal phase durations in seconds. Default is
    (np.inf, np.inf)
min_peak_heights
    Optional. Minimal peak values to be reached in both phases. Default is
    (-np.inf, -np.inf).
max_peak_heights
    Optional. Maximal peak values to be reached in both phases. Default is
    (np.inf, np.inf).

Returns
-------

TimeSeries
    A copy of `ts` with the events added.\n\n---\n\n### `most_repeatable_cycles(data: kineticstoolkit.typing_.ArrayLike, /) -> list[int]`\n\nGet the indexes of the most repeatable cycles in an array.

This function returns an ordered list of the most repeatable to the least
repeatable cycles.

It works by recursively discarding the cycle that maximizes the
root-mean-square error between the cycle and the average of every
remaining cycle, until there are only two cycles remaining. The function
returns a list that is the reverse order of cycle removal: first the two
last cycles, then the last-removed cycle, and so on. If two cycles are
equivalently repeatable, they are returned in order of appearance.

Cycles that include at least one NaN are excluded.

Parameters
----------

data
    Stacked time-normalized data to analyze, in the shape
    (n_cycles, n_points).

Returns
-------

list[int]
    List of indexes corresponding to the cycles in most to least
    repeatable order.

Example
-------

>>> import kineticstoolkit.lab as ktk
>>> import numpy as np
>>>
>>> # Create a data sample with four different cycles, the most different
>>>
>>> # being cycle 2 (cos instead of sin), then cycle 0.
>>>
>>> x = np.arange(0, 10, 0.1)
>>> data = np.array([np.sin(x),         np.sin(x) + 0.14,         np.cos(x) + 0.14,         np.sin(x) + 0.15])
>>>
>>

>>> ktk.cycles.most_repeatable_cycles(data)
>>> [1, 3, 0, 2]\n\n---\n\n### `stack(ts: kineticstoolkit.timeseries.TimeSeries, *, n_points: int = 100) -> dict[str, numpy.ndarray]`\n\nStack time-normalized TimeSeries data into a dict of arrays.
>>>
>>

This method returns the data of a time-normalized TimeSeries as a dict
where each key corresponds to a TimeSeries data key, and contains a numpy
array where the first dimension is the cycle, the second dimension is the
percentage of the cycle, and the other dimensions are the data itself.

Parameters
----------

ts
    The time-normalized TimeSeries.
n_points
    Optional. The number of points the TimeSeries has been time-normalized
    on.

Returns
-------

dict[str, np.ndarray]

See Also
--------

ktk.cycles.unstack\n\n---\n\n### `time_normalize(ts: kineticstoolkit.timeseries.TimeSeries, event_name1: str, event_name2: str, *, n_points: int = 100, span: list[int] | None = None) -> kineticstoolkit.timeseries.TimeSeries`\n\nTime-normalize cycles in a TimeSeries.

This method time-normalizes the TimeSeries at each cycle defined by
event_name1 and event_name2 on n_points. The time-normalized cycles are
put end to end. For example, for a TimeSeries that contains three
cycles, a time normalization with 100 points will give a TimeSeries
of length 300. The TimeSeries' events are also time-normalized, including
event_name1 but with event_name2 renamed as "_".

Parameters
----------

ts
    The TimeSeries to analyze.
event_name1
    The event name that corresponds to the beginning of a cycle.
event_name2
    The event name that corresponds to the end of a cycle.
n_points
    Optional. The number of points of the output TimeSeries.
span
    Optional. Specifies which normalized points to include in the output
    TimeSeries. See note below.

Returns
-------

TimeSeries
    A new TimeSeries where each cycle has been time-normalized.

Warning
-------

The span argument is experimental and was introduced in version 0.4.
**The following behavior may change in the future**. Don't rely on it in
long-term scripts for now. You can use it to define which normalized
points to include in the output TimeSeries. For example, to normalize in
percents and to include only data from 10 to 90% of each cycle, assign
100 to n_points and [10, 90] to span. The resulting TimeSeries will then
be expressed in percents and wrap each 80 points. It is also possible to
include pre-cycle or post-cycle data. For example, to normalize in
percents and to include 20% pre-cycle and 15% post-cycle, assign 100 to
n_points and [-20, 15] to span. The resulting TimeSeries will then wrap
each 135 points with the cycles starting at 20, 155, etc. and ending at
119, 254, etc. For each cycle, events outside the 0-100% spans are ignored.\n\n---\n\n### `unstack(data: dict[str, numpy.ndarray], /) -> kineticstoolkit.timeseries.TimeSeries`\n\nUnstack time-normalized data from a dict of arrays to a TimeSeries.

This method creates a time-normalized TimeSeries by putting each cycle
from the provided data dictionary end to end.

Parameters
----------

data
    A dict where each key contains a numpy array where the first dimension
    is the cycle, the second dimension is the percentage of the cycle, and
    the other dimensions are the data itself.

Returns
-------

TimeSeries

See Also
--------

ktk.cycles.stack\n\n---\n\n## Module: `kineticstoolkit.tools`\n\nProvide miscellaneous helper functions.\n\n### **Functions**\n\n### `change_defaults(change_ipython_dict_repr: bool = True, change_matplotlib_defaults: bool = True, change_numpy_print_options: bool = True, change_warnings_format: bool = True) -> None`\n\nEnable Kinetics Toolkit's lab goodies.

This function does not affect Kinetics Toolkit's inner workings. It exists
mostly for cosmetic reasons, so that working with ktk in an IPython console
(e.g., Spyder, Jupyter) is more enjoyable. It changes IPython, Matplotlib,
and numpy's defaults for the current session only. The usual way to call
it is right after importing Kinetics Toolkit.

Parameters
----------

change_ipython_dict_repr
    Optional. True to summarize default dictionary printouts in IPython. When
    False, dictionary printouts look like::

    {'data1': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                         17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
         'data2': array([  0,   1,   4,   9,  16,  25,  36,  49,  64,  81, 100, 121, 144,
                         169, 196, 225, 256, 289, 324, 361, 400, 441, 484, 529, 576, 625,
                          676, 729, 784, 841])}

    When True, dictionary printouts look like::

    {
            'data1': <array of shape (30,)>
            'data2': <array of shape (30,)>
        }

change_matplotlib_defaults
    Optional. True to change default figure size, autolayout, dpi, line
    width, and colour order in Matplotlib. The dpi and figure size are
    optimized for interactive work in default Matplotlib figures.
    Additionally, the default colour order is changed to (rgbcmyko).
    The first colours, (rgb), are consistent with the colours assigned to
    x, y, and z in most 3D visualization software.

change_numpy_print_options
    Optional. True to change default print options in numpy to use fixed
    point notation in printouts.

change_warnings_format
    Optional. True to change the warnings module's default to a more extended
    format with file and line number.

Returns
-------

None

Note
----

This function is called automatically when importing Kinetics Toolkit in
lab mode::

    import kineticstoolkit.lab as ktk\n\n---\n\n###`check_interactive_backend() -> None`\n\nWarn if Matplotlib is not using an interactive backend.

To disable these warnings, for instance if we are generating
documentation and we need the Player to show a figure, set
ktk.config.interactive_backend_warning to False.\n\n---\n\n## Module: `kineticstoolkit.player`\n\nProvides the Player class to visualize points and frames in 3d.

The Player class is accessible directly from the toplevel Kinetics
Toolkit namespace (i.e., ktk.Player).\n\n### **Classes**\n\n### `Player(*ts: kineticstoolkit.timeseries.TimeSeries, interconnections: dict[str, dict[str, typing.Any]] = {'ForcePlatforms': {'Links': [['*_Corner1', '*_Corner2'], ['*_Corner2', '*_Corner3'], ['*_Corner3', '*_Corner4'], ['*_Corner1', '*_Corner4']], 'Color': (0.5, 0.0, 1.0)}}, vectors: dict[str, dict[str, typing.Any]] = {'*Force': {'Origin': '*COP', 'Scale': 0.001, 'Color': (1.0, 1.0, 0.0)}}, current_index: int = 0, current_time: float | None = None, playback_speed: float = 1.0, up: str = 'y', anterior: str = 'x', zoom: float = 1.0, azimuth: float = 0.0, elevation: float = 0.2, pan: tuple[float, float] = (0.0, 0.0), target: tuple[float, float, float] = (0.0, 0.0, 0.0), perspective: bool = True, track: bool = False, default_point_color: str | tuple[float, float, float] = (0.8, 0.8, 0.8), default_interconnection_color: str | tuple[float, float, float] = (0.8, 0.8, 0.8), default_vector_color: str | tuple[float, float, float] = (1.0, 1.0, 0.0), point_size: float = 4.0, interconnection_width: float = 1.5, vector_width: float = 2.0, frame_size: float = 0.1, frame_width: float = 3.0, grid_size: float = 10.0, grid_subdivision_size: float = 1.0, grid_width: float = 1.0, grid_origin: tuple[float, float, float] = (0.0, 0.0, 0.0), grid_color: str | tuple[float, float, float] = (0.3, 0.3, 0.3), background_color: str | tuple[float, float, float] = (0.0, 0.0, 0.0), **kwargs)`\n\nA class that allows visualizing points and frames in 3D.

`player = ktk.Player(parameters)` creates and launches an interactive
Player instance. Once the window is open, press `h` to show a help
overlay.

All of the following parameters are also accessible as read/write
properties, except the contents and the interconnections that are
accessible using `get_contents`, `set_contents`, `get_interconnections`
and `set_interconnections`.

Parameters
----------

*ts
    Contains the points and frames to visualize, where each data
    key is either a point position expressed as Nx4 array, or a frame
    expressed as a Nx4x4 array. Multiple TimeSeries can be provided.

interconnections
    Optional. Each key corresponds to a group of interconnections, which
    is a dictionary with the following keys:

    - "Links": list of connections where each string is a point
      name. For example, to create a link that connects Point1 to Point2,
      and another link that spans Point3, Point4 and Point5::

    interconnections["Example"]["Links"] = [
              ["Point1", "Point2"],
              ["Point3", "Point4", "Point5"]
          ]

    which internally is converted to::

    interconnections["Example"]["Links"] = [
              ["Point1", "Point2"],
              ["Point3", "Point4"],
              ["Point4", "Point5"]
          ]

    Point names can include wildcards (*) either as a prefix or as a
      suffix. This is useful to apply a single set of interconnections to
      multiple bodies. For instance, if the Player's contents include
      these points: [Body1_HipR, Body1_HipL, Body1_L5S1, Body2_HipR,
      Body2_HipL, Body2_L5S1], we could link L5S1 and both hips at once
      using::

    interconnections["Pelvis"]["Links"] = [
              ["*_HipR", "*_HipL", "*_L5S1"]
          ]

    - "Color": character or tuple (RGB) that represents the color of the
      link. These two examples are equivalent::

    interconnections["Pelvis"]["Color"] = 'r'
          interconnections["Pelvis"]["Color"] = (1.0, 0.0, 0.0)

    Its default value connects the four corners of force platforms in
    purple::

    interconnections = {
            "ForcePlatforms": {
                "Links": [['*_Corner1', '*_Corner2'],
                          ['*_Corner2', '*_Corner3'],
                          ['*_Corner3', '*_Corner4'],
                          ['*_Corner1', '*_Corner4']]
                "Color": (0.5, 0.0, 1.0)
            }
        }

vectors
    Optional. A dictionary where each key is the name of a vector and each value
    contains its origin, scale, and color. For example::

    vectors = {
            "WristForce": {
                "Origin": "WristCenter",
                "Scale": 0.001,
                "Color": (1.0, 1.0, 0.0)
            },
            "ElbowForce": {
                "Origin": "ElbowCenter",
                "Scale": 0.001,
                "Color": (1.0, 1.0, 0.0)
            },
        }

    will draw lines for the forces WristForce and ElbowForce, with their
    origin being at WristCenter and ElbowCenter, and with a scale of 0.001
    metre per newton. Force and point names can include wildcards (*)
    either as a prefix or as a suffix. For instance, to draw forces
    recorded by multiple force plates, we could use::

    vectors = {
            "*Force": {
                "Origin": "*COP",
                "Scale": 0.001,
                "Color": (1.0, 1.0, 0.0)
            }
        }

    which would assign any point ending by "COP" to its counterpart force.
    This is the default, so that force plate data read by read_c3d_file
    are shown by default in the Player.

current_index
    Optional. The current index being shown.

current_time
    Optional. The current time being shown.

playback_speed
    Optional. Speed multiplier. Set to 1.0 for normal speed, 1.5 to
    increase playback speed by 50%, etc.
up
    Optional. Defines the ground plane by setting which axis is up. May be
    {"x", "y", "z", "-x", "-y", "-z"}. Default is "y".

anterior
    Optional. Defines the anterior direction. May be
    {"x", "y", "z", "-x", "-y", "-z"}. Default is "x".

zoom
    Optional. Camera zoom multiplier.

azimuth
    Optional. Camera azimuth in radians. If `anterior` is set, then an
    azimuth of 0 corresponds to the right sagittal plane, pi/2 to the
    front frontal plane, -pi/2 to the back frontal plane, etc.

elevation
    Optional. Camera elevation in radians. Default is 0.2. If `up` is set,
    then a value of 0 corresponds to a purely horizontal view, pi/2 to the
    top transverse plane, -pi/2 to the bottom transverse plane, etc.

perspective
    Optional. True to draw the scene using perspective, False to draw the
    scene orthogonally.

pan
    Optional. Camera translation (panning). Default is (0.0, 0.0).

target
    Optional. Camera target in metres. Default is (0.0, 0.0, 0.0).

track
    Optional. False to keep the camera static, True to follow the last
    selected point when changing index. Default is False.

default_point_color
    Optional. Default color for points that do not have a "Color"
    data_info. Can be a character or tuple (RGB) where each RGB color is
    between 0.0 and 1.0. Default is (0.8, 0.8, 0.8).

default_interconnection_color
    Optional. Default color for interconnections. Can be a character or
    tuple (RGB) where each RGB color is between 0.0 and 1.0. Default is
    (0.8, 0.8, 0.8).

default_vector_color
    Optional. Default color for vectors. Can be a character or tuple (RGB)
    where each RGB color is between 0.0 and 1.0. Default is
    (1.0, 1.0, 0.0).

point_size
    Optional. Point size as defined by Matplotlib marker size. Default is
    4.0.

interconnection_width
    Optional. Width of the interconnections as defined by Matplotlib line
    width. Default is 1.5.

vector_width
    Optional. Width of the vectors as defined by Matplotlib line
    width. Default is 2.0.

frame_size
    Optional. Length of the frame axes in metres. Default is 0.1.

frame_width
    Optional. Width of the frame axes as defined by Matplotlib line width.
    Default is 3.0.

grid_size
    Optional. Length of one side of the grid in metres. Default is 10.0.

grid_subdivision_size
    Optional. Length of one subdivision of the grid in metres. Default is
    1.0.

grid_width
    Optional. Width of the grid lines as defined by Matplotlib line width.
    Default is 1.0.

grid_origin
    Optional. Origin of the grid in metres. Default is (0.0, 0.0, 0.0).

grid_color
    Optional. Color of the grid. Can be a character or tuple (RGB) where
    each RGB color is between 0.0 and 1.0. Default is (0.3, 0.3, 0.3).

background_color
    Optional. Background color. Can be a character or tuple (RGB) where
    each RGB color is between 0.0 and 1.0. Default is (0.0, 0.0, 0.0).

Note
----

Matplotlib must be in interactive mode.\n\n---\n\n## Module: `kineticstoolkit.gui`\n\nProvide simple GUI functions.

Warning
-------

This module is private and should be considered only as helper functions
for Kinetics Toolkit's own use.\n\n### **Functions**\n\n### `button_dialog(message: str = 'Please select an option.', choices: list[str] = ['Cancel', 'OK'], **kwargs) -> int`\n\nCreate a blocking dialog message window with a selection of buttons.

Parameters
----------

message
    Message that is presented to the user.
choices
    List of button text.

Returns
-------

int
    The button number (0 = First button, 1 = Second button, etc.). If the
    user closes the window instead of clicking a button, a value of -1 is
    returned.\n\n---\n\n### `get_credentials() -> tuple[str, str]`\n\nAsk the user's username and password.

Returns
-------

tuple[str]
    A tuple of two strings containing the username and password,
    respectively, or an empty tuple if the user closed the window.\n\n---\n\n### `get_filename(initial_folder: str = '.') -> str`\n\nGet file name interactively using a file dialog window.

Parameters
----------

initial_folder
    Optional. The initial folder of the file dialog.

Returns
-------

str
    The full path of the selected file. An empty string is returned if the
    user cancelled.\n\n---\n\n### `get_folder(initial_folder: str = '.') -> str`\n\nGet folder interactively using a file dialog window.

Parameters
----------

initial_folder
    Optional. The initial folder of the file dialog.

Returns
-------

str
    The full path of the selected folder. An empty string is returned if
    the user cancelled.\n\n---\n\n### `message(message: str = '', **kwargs) -> None`\n\nShow a message window.

Parameters
----------

message
    The message to show. Use '' to close every message window.\n\n---\n\n### `set_color_order(setting: str | list[typing.Any]) -> None`\n\nDefine the standard color order for matplotlib.

Parameters
----------

setting
    Either a string or a list of colors.

    - If a string, it can be either:
        - 'default': Default v2.0 matplotlib colors.
        - 'classic': Default classic Matlab colors (bgrcmyk).
        - 'xyz': Same as classic but begins with rgb instead of bgr to
           be consistent with most 3D visualization software.

    - If a list, it can be either a list of chars from [bgrcmyk], a list of
      hexadecimal color values, or any list supported by matplotlib's
      axes.prop_cycle rcParam.\n\n---\n\n## Module:`kineticstoolkit.config`\n\nProvide configuration values for Kinetics Toolkit's internal workings.\n\n## Module: `kineticstoolkit.exceptions`\n\nProvide functions related to exceptions.

For internal use only.\n\n### **Classes**\n\n### `TimeSeriesEventNotFoundError()`\n\nThe requested event occurrence was not found.\n\n---\n\n### `TimeSeriesMergeConflictError()`\n\nBoth TimeSeries have a same data key.\n\n---\n\n### `TimeSeriesRangeError()`\n\nThe requested operation exceeds the TimeSeries' time range.\n\n---\n\n### **Functions**\n\n### `raise_ktk_error(e) -> None`\n\nRe-raise an exception with a user message on how to report this bug.\n\n---\n\n### `warn_once(message: str, category=<class 'UserWarning'>, stacklevel: int = 1) -> None`\n\nRaise a warning only once.\n\n---\n\n
