=====
Usage
=====

To use SOSAT in a project::

    from SOSAT import stress_state


    ss = StressState(1.0,
                     2.5,
                     0.3,
                     depth_unit='km',
                     density_unit='g/cm^3',
                     pressure_unit='MPa')

    sigv = ss.vertical_stress