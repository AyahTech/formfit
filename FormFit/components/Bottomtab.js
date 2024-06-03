import * as React from 'react';
import { Ionicons } from '@expo/vector-icons';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useRoute } from '@react-navigation/native';
import { Image } from 'react-native';
import Exercise from '../Pages/Exercise';
import AskCoach from '../Pages/AskCoach';
import ImproveForm from '../Pages/ImproveForm';

const Tab = createBottomTabNavigator();

// Custom component for the header logo
function HeaderLogo() {
  return (
    <Image
      source={require('../assets/images/Logo.png')} // replace with your logo's path
      style={{ width: 150, height: 90 , marginLeft: -20 }} // adjust size as needed
      resizeMode="contain"
    />
  );
}

function BottomTab() {
  const route = useRoute();

  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color }) => {
          let iconName;

          if (route.name === 'Exercise') {
            iconName = focused ? 'menu' : 'menu-outline';
          } else if (route.name === 'Ask Coach') {
            iconName = focused ? 'chatbubble-sharp' : 'chatbubble-outline';
          } else if (route.name === 'Improve Form') {
            iconName = focused ? 'aperture' : 'aperture-outline';
          }

          return <Ionicons name={iconName} size={30} color={color} />;
        },
        tabBarActiveTintColor: 'blue',
        tabBarInactiveTintColor: 'gray',
        tabBarStyle: { 
          height: 100, 
          paddingBottom: 20, 
          display: getTabBarVisibility(route) ? 'flex' : 'none', // Conditionally hide the tab bar
        },
        tabBarLabelStyle: { 
          paddingBottom: 10, 
          fontWeight: '600', 
          fontSize: 12 
        }, // Adjust the label position
      })}
      initialRouteName="Exercise"
    >
      <Tab.Screen 
        name="Exercise" 
        component={Exercise} 
        options={{ 
          headerShown: true, 
          headerTitle: () => <HeaderLogo />, // Use custom logo component
          headerTitleAlign: 'start', 
        }} 
      />
      <Tab.Screen 
        name="Ask Coach" 
        component={AskCoach} 
        options={{ headerShown: false }} 
      />
      <Tab.Screen 
        name="Improve Form" 
        component={ImproveForm} 
        options={{ headerShown: false }} 
      />
    </Tab.Navigator>
  );
}

// Helper function to determine tab bar visibility
function getTabBarVisibility(route) {
  const routeName = route.name;
  if (routeName === 'Ask Coach') {
    return false;
  }
  if (routeName === 'Improve Form') {
    return false;
  }
  return true;
}

export default BottomTab;
