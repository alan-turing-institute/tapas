# Data format

Prive presently deals with tabular data. It consumes and produces the following
tabular datasets:

- The user-supplied raw data;
- Any input to the user-supplied privacy-enhancing method;
- Any output from the user-supplied privacy-enhancing method.

A _table_ is:

1. A tuple of _field names_ (sometimes called ‘column headers’);
2. For each field name, a _type_ and a _representation_;
3. A finite set of tuples (also called ‘rows’), each element of which is a
   representation of the type in the coresponding field name.
   
In prive, the set of tuples must be given as a comma-separated-values file
whilst the first two items are declared in a separate JSON file, called the
‘table schema’ for short.

This document describes the types of data understood by prive along with their
representations.

## JSON format

A table schema is an array of field descriptions. The order matters, and should
match the order of columns in the csv file. (The csv file should _not_ include a
header row.)

A field description is an object with the following elements:
- `name`
- `type`
- `representation`

## The meaning of types

In this context, a type is a set (the set of possible values of that type) plus
possibly some additional structure on the set. It is not entirely clear which
types should be taken as primitive. The situation is most unclear for infinite
types; for finite types we don't expect much disagreement.

Broadly, types can be finite, countably-infinite, or continuous; and for each we
might have the additional structure of a total order. If we do have a total
order then, for the infinite types, there is a question of whether there is no
upper or lower bound; a least element; or both a least and a greatest
element. Finally one might ask whether the set is dense in the order (ie,
whether for any two elements there is one lying between them) and one might also
ask whether there order is complete (in the sense that every subset that is
bounded above has a least upper bound).

(One principled reason for choosing these particular types --- at least, the
infinite, ordered ones --- is that they are _initial_ of their type. At least,
it would be a good reason if it were true.)

### Finite types

#### `finite`

A finite, unordered set. 

There are two representations: either an integer or an array of string. If the
representation is an integer, N, then the possible values are the integers
from 0 to N inclusive. If an array, then the elements of the array enumerate
the possible values.

#### `finite/ordered`

A finite, totally ordered set. The allowed representations are the same the
`finite` type except that now the order is that given by the natural order on
the integers between 0 and N or the order of the array of strings.

### Countably-infinite types

#### `countable`

A countably-infinite, unordered set. The allowed values of `representation` are either
`"integer"` or `"string"`. 

#### `countable/ordered`

A countably-infinite, totally-ordered set, with neither an upper nor a lower
bound. There is only one allowed representation, which is the integers (positive
and negative), so no `representation` field is required.

#### `countable/ordered/least`

A countably-infinite, totally-ordered set with a least element. 


### Uncountably-infinite types





## Future extentions

`countable/partial`? (Strings with prefix orderg!)



