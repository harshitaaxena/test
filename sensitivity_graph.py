from library import *

def sensitivity_graph(data, input_data_, coefficients, intercept, ub_lb_df, p):
    coefficients_dict = dict(zip(list(input_data_.columns), list(np.append(coefficients, [intercept]))))
    # Convert values to lists
    coefficients_dict = {key: [value] for key, value in coefficients_dict.items()}
    coefficients_df = pd.DataFrame(coefficients_dict)
    a = np.dot(np.array(input_data_.drop(p, axis=1))[0], np.array(coefficients_df.drop(p, axis=1))[0])
    slope = float(coefficients_df[p])
    intercept = float(a)

    # Define the equation
    def equation(x, slope, intercept):
        return intercept + slope * x

    x = np.arange(int(ub_lb_df[ub_lb_df['Controllable_Parameter'] == p].lower),
                  int(ub_lb_df[ub_lb_df['Controllable_Parameter'] == p].upper), 0.1)
    y = equation(x, slope, intercept)

    # Dynamic user input for x
    user_x = float(input_data_[p])
    # Calculate y for the user input x
    user_y = equation(user_x, slope, intercept)

    # Plot the equation
    # Plot the user input point
    fig, ax = plt.subplots()
    ax.plot(user_x, user_y, 'ro')  # 'ro' for red circles
    ax.annotate(f'({round(user_x, 2)}, {round(user_y, 2)})', (user_x, user_y), xytext=(-25, 15),
                textcoords='offset points', arrowprops=dict(arrowstyle='->'))
    ax.plot(x, y)

    if p =='Air Velocity':
        ax.set_xlabel('Steam Velocity (m/s)')
    elif p=='Steam Inlet Temp':
        ax.set_xlabel('Steam Inlet Temp (C)')
    elif p== 'Steam Outlet Temp':
         ax.set_xlabel('Steam Outlet Temp (C)')
    elif p== 'Steam Sep Line Temp':
         ax.set_xlabel('Steam Sep Line Temp (C)')
    elif p== 'Transfer cool air temp':
        ax.set_xlabel('Cool Air (C)')
    elif p== 'Transfer Air':
        ax.set_xlabel('Transfer Air (C)')
    elif p== 'Steam Pressure':
        ax.set_xlabel('Steam Pressure (bar)')
    elif p== 'Ambient Temp':
        ax.set_xlabel('Ambient Temp (F)')
    elif p== 'Ambient Humidity':
        ax.set_xlabel('Ambient Humidity (%)')
    elif p== 'Feed_rate':
        ax.set_xlabel('Feed Rate (kg/hr)')
    elif p== 'LOD_raw':
        ax.set_xlabel('Raw Husk LOD %')
    elif p== 'Swell_vol':
        ax.set_xlabel('Swell Volume (ml)')
    elif p== 'on_30':
        ax.set_xlabel('% on 30 mesh')
    elif p== 'thru_70':
        ax.set_xlabel('% thru 70 mesh')
    else:
        ax.set_xlabel('% thru 100 mesh')
        
    ax.set_ylabel('LOD %')
    ax.grid(True)
    st.pyplot(fig)
    