private List<Filter> getFilters(HttpServletRequest request){
    for(SecurityFilterChain chain: filterChains){
        if(chain.matches(request)){
            return chain.getFilters();
        }
    }
    return null;
}