#pragma once
template<unsigned int VALUE>
struct CE_LOG2{
	static const unsigned int V = CE_LOG2<(VALUE >> 1)>::V + 1;
};
template<>
struct CE_LOG2<1>{
	static const unsigned int V = 1;
};
template<>
struct CE_LOG2<0>{
	static const unsigned int V = 0;
};
