===========
Data format
===========

``TAPAS`` presently deals with tabular data. It consumes and produces the
following tabular datasets:

-  The user-supplied raw data;
-  Any input to the user-supplied privacy-enhancing method;
-  Any output from the user-supplied privacy-enhancing method.

In order that ``TAPAS`` can interpret the data given to it and to ensure
that it produces data that is a valid input to the user-supplied
privacy-enhancing method, we need a method of describing tabular data.

The *format* for tabular data used by ``TAPAS`` is a csv file, where each
line is a comma-separated tuple of values, and the values in the
corresponding position in different rows have the same ‘type’. The main
challenge is describing the \`type’ of each field in the data as well as
how to interpret the representation of the type used in the table.

The format we have chosen for storing this metadata is a separate json
file, the “table schema” (not to be confused with the JSON schema which
describes the format of any table schema).

This document describes the types of data understood by ``TAPAS`` along with
their representations.

JSON format
-----------

A table schema is an array of field descriptions. The order matters, and
should match the order of columns in the csv file. (The csv file should
*not* include a header row.)

A field description is an object with the following elements: - ``name``
(an arbitrary string) - ``type`` - ``representation``

The meaning of types
--------------------

In this context, a type is a set (the set of possible values of that
type) plus possibly some additional structure on the set. It is not
entirely clear which types should be taken as primitive. The situation
is most unclear for infinite types; for finite types we don’t expect
much disagreement.

For most types, the additional structure is whether or not the set has a
total order and, if it does, whether there is a least element, or both a
least and a greatest element.

The distinction between ``type`` and ``representation`` is not quite
right. For example, perhaps ``date`` ought to be its own type (which
happens to be “isomorphic to” ``countable/ordered``) with its own
representation. For now, our schema makes ``date`` merely a way of
representing ``countable/ordered``.

Likewise, we’ve decided to call the continuous types “real”, as there is
additional structure, beyond simply the order, which is commonly assumed
(ie, addition and multiplication). In addition, the integers (which are
countable) have a notion of “next element”, which is not true of strings
(which are also countable) but we have ignored this structure. Our
scheme is therefore somewhat inconsistent – or, at least, incomplete –
in how it thinks of types.

(One principled reason for choosing these particular types — at least,
the infinite, ordered ones — is that they are *initial* of their type,
in the category-theory sense. At least, it would be a good reason if it
were true.)

+-----------------------------+--------------------+-------------------------------+
| ``type``                    |``representation``  | Meaning                       |
+=============================+====================+===============================+
| ``"finite"``                |An integer, N       | 0, 1, 2, …, N - 1             |
+-----------------------------+--------------------+-------------------------------+
|                             |An array of strings | The given strings             |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
| ``"finite/ordered"``        |An integer, N       | 0, 1, 2, …, N - 1             |
+-----------------------------+--------------------+-------------------------------+
|                             |An array of strings | The given strings, in the     |
|                             |                    | given order                   |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
| ``"countable"``             |``"integer"``       | 0, 1, 2, …,                   |
+-----------------------------+--------------------+-------------------------------+
|                             |``"string"``        | Any string                    |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
|``"countable/ordered"``      |``"integer"``       | …, -2, -1, 0, 1, 2, …         |
+-----------------------------+--------------------+-------------------------------+
|                             |``"date"``          | YYYY-MM-DD or YYYYMMDD        |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
|``"countable/ordered/least"``|``"integer"``       | 0, 1, 2, …                    |
+-----------------------------+--------------------+-------------------------------+
|                             |``"string"``        | Strings, with dictionary      |
|                             |                    | order                         |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
| ``"real"``                  |``"number"``        | Any decimal approximation     |
+-----------------------------+--------------------+-------------------------------+
|                             |``"datetime"``      | YYYY-MM-DDThh:mm:ss.sss, or   |
+-----------------------------+--------------------+-------------------------------+
|                             |                    | YYYYMNMDDThhmmss.sss          |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
| ``"real/non-negative"``     |``"number"``        | Any decimal approximation     |
+-----------------------------+--------------------+-------------------------------+
|                             |                    |                               |
+-----------------------------+--------------------+-------------------------------+
| ``"interval"``              |``"number"``        | The closed interval [0, 1].   |
+-----------------------------+--------------------+-------------------------------+

Future extensions
-----------------

- ``countable/partial``? (Strings with prefix order!)
- ``countable/ordered/dense`` (Decimals) 
- ``countable/ordered/least/dense`` (Decimals or strings with dictionary order)
