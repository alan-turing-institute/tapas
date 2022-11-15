.. tapas documentation master file, created by
   sphinx-quickstart on Thu Apr 14 11:48:41 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TAPAS's documentation!
=================================

**TAPAS** is a Python library for evaluating the privacy of synthetic data from
an adversarial perspective.

.. note::
   This project is under active development.

   Thus, the API of each of the modules could change at any time.

   Finally, we welcome contributions to our package in any way. 


This library is designed to be an extensible privacy toolbox which should be
accessible to: 

- Developers or users of synthetic data generators who want to test their
  generator against a range of known attacks and a diversity of threat models.
- Privacy researchers who want to test and develop new attacks against
  synthetic data generators.

``TAPAS`` implements a panoply of diverse attacks in order to extract private
information about real datasets from synthetic datasets. Importantly, should
no attack be found to succeed, it does not mean that the generator is safe, as
more sophisticated/specifically tailored attacks might exist. This package thus
mostly aims at probing implementations for known vulnerabilities.


Using the Package 
-----------------

To meet the two use cases above we provide two different interfaces into the package. 
The first is a pure Python interface which can be combined directly with
standard python pipelines, see the ``/examples`` folder.

The second is a purely command line interface using the ``tapas`` command which
is directly installed when you set up the package. This interface allows you to
interact with our package without having to develop in a python ecosystem.
*Warning: this is currently unsupported -- use Python instead*


API
--------

.. autosummary::
   :toctree: _autosummary
   :template: custom-module-template.rst
   :recursive:

   tapas

Contents
--------

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   dataset-schema
   modelling-threats
   library-of-attacks
   implementing-attacks
   evaluation



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
