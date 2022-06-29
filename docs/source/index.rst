.. prive documentation master file, created by
   sphinx-quickstart on Thu Apr 14 11:48:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to prive's documentation!
=================================

**prive** is a Python library for evaluating the privacy of synthetic data.

.. note::
   This project is under active development.

This library is designed to be an extensible privacy toolbox which should be
accessible to: 

- Privacy researchers who may want to test and develop me new attacks against
  synthetic data 
- Users of synthetic data generators to test their generated synthetic data
  using said attacks.  

We must highlight that an attack failing to find any problems does not mean
that the data is necessarily safe (i.e. a more sophisticated attack may be
successful), however if an attack identifies issues under a realistic threat
model then the level of privacy in your dataset is likely not at the desired
level. 



## Using the Package 

To meet the two use cases above we provide two different interfaces into the package. 
The first is a pure python interface which can be combined directly with
standard python pipelines, see the example section.

The second is a purely command line interface using the `prive` command which
is directly installed when you set up the package. This interface allows you to
interact with our package without having to develop in a python ecosystem.




Contents
--------
.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   prive

.. toctree::
   dataset-schema



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
