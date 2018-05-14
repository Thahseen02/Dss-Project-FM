tic();
exec("data.sce");
//initialization
iterations=1;
first=1;
b=0;
while iterations <= max_iter
//seed value is not set yet
///stage 1 :addition of excess rakes to original demand
surplus = total_rakes - sum(demand);
available_space = storage_capacity - initial_stock_level;
dv1=1+9*rand(1,1);  //random number between 1 and 10 
while surplus>0 && dv1 < 6
        available = (min (max_allottment,available_space))- demand;
        ratio = (storage_capacity./(initial_stock_level+demand));
        max_ratio = max(ratio);
        f1 = find(ratio == max_ratio);
        max_allo = min (available(f1(1)),surplus);
        allo = min(2,max_allo);
        rn2 = floor(allo*rand(1,1))+1;; //selecting a random integer
        surplus = surplus - rn2;
        demand(f1(1))= demand(f1(1))+rn2;
    end
while surplus>0 && dv1 > 5
        available = (min (max_allottment,available_space))- demand;         
        f1 = find (available>0);
        size_f1 = size(f1);
        rn1 = floor(size_f1*rand(1,1))+1; //selecting a random integer
        r_depot = f1(rn1);
        max_allo = min (available(r_depot),surplus);
        allo = min(2,max_allo);
        rn2 = floor(allo*rand(1,1))+1; //selecting a random integer
        surplus = surplus - rn2;
        demand(r_depot)= demand(r_depot)+rn2;
    end

/////stage 2 
weekly_distribution=zeros(total_warehouses,total_weeks);  //initializing weekly distribution
comb = zeros(total_warehouses,total_warehouses,total_weeks); //initilaizing combination matrix
monthly_allotment = demand;
sum1 = sum(weekly_distribution);
sum2 = sum(sum1);
i=1;

//////////////allocation of full rakes to warehouses with higher demand
excess = find(demand>4);
num = size(excess);
num_excess = num(1);
for q = 1:num_excess
        higher_demand_q=excess(q);
        excess_qty_q = demand(higher_demand_q)-4;
        for n=1:excess_qty_q
            selected_week_f1_n = floor(total_weeks*rand(1,1))+1;
            if weekly_distribution(higher_demand_q,selected_week_f1_n)==0 
                in=2;
                weekly_distribution(higher_demand_q,selected_week_f1_n)= weekly_distribution(higher_demand_q,selected_week_f1_n)+in;
                sum2=sum2+in;
            end      
        end     
    end
//////////stage 3:
////////////////half and full rake allocation to warehouses
while sum2 < sum(demand)
        rand_depot = floor(total_warehouses*rand(1,1))+1;
        rand_week = floor(total_weeks*rand(1,1))+1;
        selected_row = weekly_distribution(rand_depot,:);
        sum3 = sum(selected_row);
        sum4 = sum(sum3);
        iteration = 1;
        dv2 = 1+9*rand(1,1);
        while (sum4<monthly_allotment(rand_depot)) && (iteration<2)               //demand not met yet
            
            if weekly_distribution(rand_depot,rand_week)==0               //if unallocated
                
                comb_row = comb_matrix(rand_depot,:);                     
                
                if sum(comb_row)> 0 && dv2<10
                   
                   
                   possible_comb = find(comb_row ==1);
                   size_find = size(possible_comb);
                   size_possible_comb = size_find(2);
                   random = floor(size_possible_comb*rand(1,1))+1;
                   selected= possible_comb(random); 
                   selected_row2=weekly_distribution(selected,:); 
                   sum5=sum(selected_row2);      
                   sum6=sum(sum5);
                   while(weekly_distribution(selected,rand_week)==0) && (sum6<monthly_allotment(selected))

                            in= 1;
                            weekly_distribution(selected,rand_week) =weekly_distribution(selected,rand_week)+in; 
                            weekly_distribution(rand_depot,rand_week)= weekly_distribution(rand_depot,rand_week)+in;
                            comb(selected,rand_depot,rand_week) = comb(selected,rand_depot,rand_week)+1; 
                            comb(rand_depot,selected,rand_week) = comb(rand_depot,selected,rand_week)+1; 
                            sum2=sum2+(2*in);
                            sum4=sum4+in;
                            sum6=sum6+in;
                           
                        end
                        count = 1;
                        
                        while ((dv2>5) && (count<2)) && (i>k_shift)
                       
                            if terminal_capacity(rand_depot)==2 && weekly_distribution(rand_depot,rand_week)==0
                           
                                c=1;
                            
                                while (sum4 < (monthly_allotment(rand_depot)-1)) && (c<2) 
                                weekly_distribution(rand_depot,rand_week)= weekly_distribution(rand_depot,rand_week)+2;
                                sum2 = sum2+2;
                                sum4 = sum4+2;
                                c=c+1;                           
                                end                               
                            end                        
                            count=count+1;                  
                        end                     
                   else if (sum(comb_row)==0)
                        
                    if terminal_capacity(rand_depot)==2
                        weekly_distribution(rand_depot,rand_week)= weekly_distribution(rand_depot,rand_week)+2
                        sum2 = sum2+2;
                        sum4 = sum4+2;
                    
                    end
                    end   
             end              
            end         
            iteration=iteration+1;           
     end
     if (pmodulo(i,k_reset)==0)
         weekly_distribution = zeros(total_warehouses,total_weeks);                           
            comb = zeros(total_warehouses,total_warehouses,total_weeks);                            
            sum2=0;                            
            excess = find(demand>4);                            
            num=size(excess);                            
            num_excess = num(1);
                                           
            for q = 1:num_excess                                    
                higher_demand_q=excess(q);                                   
                excess_qty_q = demand(higher_demand_q)-4;
                                                    
                for n=1:excess_qty_q                                           
                    selected_week_f1_n = floor(total_weeks*rand(1,1))+1;
                                                              
                    if weekly_distribution(higher_demand_q,selected_week_f1_n)==0                                                    
                        in=2;                       
                        weekly_distribution(higher_demand_q,selected_week_f1_n)= weekly_distribution(higher_demand_q,selected_week_f1_n)+in;                                                    
                        sum2=sum2+in;                                      
                    end                    
                end               
            end           
        end
        if i > k_terminate
            b=b+1;                              
            break                   
        end        
        i=i+1;        
    end
    ///stage 5
    ///calculation of penalty values 
    half_rakes = find (weekly_distribution==1);      
    full_rakes = find(weekly_distribution==2);
    size1 = size(half_rakes);
    size2 = size(full_rakes);
    cost1 = size1(1)*rake_penalty_h;
    cost2 = size2(1)*rake_penalty_f;
    rake_penalty = cost1+cost2;                     
    
    calc_matrix = ceil(weekly_distribution./2);
    cost3 = calc_matrix.*weekly_penalty;
    cost4 = sum(cost3);
    cost5 = sum(cost4);
    week_penalty = cost5;
    
    weekly_distribution_t = weekly_distribution';
    qty1 = sum(weekly_distribution_t);
    distributed_qty = qty1';
    final_stock_level = initial_stock_level + distributed_qty;
    space_utilization = (storage_capacity./final_stock_level);
    space_utilization_rounded = floor(space_utilization.*10);
    capacity_utilization_penalty = sum(space_utilization_rounded);
    total_penalty = rake_penalty + week_penalty +capacity_utilization_penalty;
    
    /////////////////detemination of the best solution
    s1 = sum (weekly_distribution);
    s2 = sum(s1);
    
    if (first == 1) && (s2 == total_rakes)      
        final_weekly_distribution = weekly_distribution;
        final_total_penalty = total_penalty;
        final_rake_penalty = rake_penalty;
        final_week_penalty = week_penalty;
        final_capacity_utilization_penalty = capacity_utilization_penalty;
        final_comb = comb;
        first=first+1;
    end
    
    if (s2 == total_rakes)&& (first ~= 1)

        if (total_penalty < final_total_penalty) 
            final_weekly_distribution = weekly_distribution;   
            final_total_penalty = total_penalty;   
            final_rake_penalty = rake_penalty;   
            final_week_penalty = week_penalty;   
            final_capacity_utilization_penalty = capacity_utilization_penalty;
            final_comb = comb;
        end  
     end      
    iterations=iterations+1;
 end
 /////displaying results
disp(demand);
disp(final_weekly_distribution);
disp(final_total_penalty);
disp(final_rake_penalty);
disp(final_week_penalty);
disp(final_capacity_utilization_penalty);
disp(final_comb);
disp(b);
t=toc();
disp(t);
