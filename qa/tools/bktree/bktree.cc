#include <string>
#include <iostream>
#include <vector>
#include "library.h"
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
using namespace boost::python;
using namespace std;

// overloadings
// vector<string>    (BKTree::*search)(string, int) = &BKTree::search;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(search_overloads, search, 2, 2)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(add_overloads, add, 1, 1)
BOOST_PYTHON_MODULE(BKTree)
{   
    class_<vector<string> >("string_vector")
        .def(vector_indexing_suite<vector<string> >());
    
    class_<BKTree>("BKTree")
        .def("add", &BKTree::add, add_overloads())
        .def("search", &BKTree::search, search_overloads());
}