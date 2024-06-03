import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, FlatList, ActivityIndicator } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useNavigation } from '@react-navigation/native';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const flatListRef = useRef(null);
  const navigation = useNavigation(); // Hook to get navigation instance

  useEffect(() => {
    console.log("Updated Messages:", messages); // Log messages whenever they update
  }, [messages]);

  const handleOptionSelect = async (option) => {
    await fetchBotResponse(option);
    setInput('');
    scrollToBottom(); // Scroll to the bottom when a new message is added
  };

  const handleSend = async () => {
    if (input.trim()) {
      const userMessage = input;
      setInput('');
      await fetchBotResponse(userMessage);
      scrollToBottom(); // Scroll to the bottom when a new message is added
    }
  };

  const fetchBotResponse = async (prompt) => {
    setMessages((prevMessages) => [...prevMessages, { type: 'user', text: prompt }]);
    setLoading(true); // Show loading indicator

    let botResponse = '';

    switch (prompt.toLowerCase()) {
      case 'what are the best exercises for diabetes?':
        botResponse = "Suicidal thoughts and self-injurious behavior have been associated with the use of GLP-1 receptor agonists, therefore, patients must consult a physician before using any GLP-1 receptor agonists for weight loss. Natural compounds such as caffeoylmalic acid, found in Urtica dioica, have shown antidiabetic properties in vitro and in vivo by stimulating glucose absorption and reducing blood glucose levels. Furthermore, it improves lipid profile, liver, and blood parameters, with moderate effects on insulin secretion. Insulin therapy remains a popular method for treating type 2 diabetes, however, patient adherence to insulin therapy can be challenging.";
        break;
      case 'what is the best exercise for cardiovascular diseases?':
        botResponse = "Summary: The best exercise for cardiovascular diseases involves a combination of aerobic exercise, resistance training, and high-intensity interval training (HIIT):\n\nAerobic Exercise: Recommended as the primary exercise type for cardiovascular health. It includes activities like walking, jogging, swimming, and cycling. Benefits include enhanced heart efficiency, lower blood pressure, and improved cholesterol levels.\n\nResistance Training: Complementary to aerobic exercise, resistance training improves muscle strength and endurance, contributing to overall cardiovascular health and aiding recovery post-cardiac events.\n\nHigh-Intensity Interval Training (HIIT): A time-efficient exercise that alternates between short bursts of intense activity and periods of rest or lower-intensity exercise. It significantly improves heart function and cardiovascular fitness.\n\nBest Recommendation:\n\nAerobic exercise is the most universally recommended for cardiovascular health due to its broad benefits and ease of implementation. Combining aerobic exercise with resistance training and incorporating HIIT can optimize cardiovascular benefits.";
        break;
      case 'how can i modify exercises for pregnancy?':
        botResponse = "Summary: When modifying exercises for pregnancy, it is essential to prioritize safety, comfort, and the specific needs of the pregnant body. Here are key guidelines and modifications for different types of exercises:\n\nGeneral Guidelines:\n\nConsult a Healthcare Provider: Always get approval from a healthcare provider before starting or continuing an exercise routine during pregnancy. Listen to Your Body: Pay attention to your body's signals. Avoid overexertion, and stop exercising if you experience pain, dizziness, or shortness of breath. Stay Hydrated: Drink plenty of water before, during, and after exercise to stay hydrated. Avoid Overheating: Exercise in a cool environment and avoid activities that cause overheating.\n\nCardiovascular Exercise:\n\nModify Intensity: Reduce the intensity and duration of aerobic exercises. Low-impact activities like walking, swimming, and stationary cycling are recommended. Avoid High-Risk Activities: Avoid exercises with a high risk of falling or abdominal trauma, such as skiing, horseback riding, and contact sports.\n\nStrength Training:\n\nReduce Weights: Use lighter weights and increase repetitions instead of lifting heavy weights. Focus on Core and Pelvic Floor: Strengthen the core and pelvic floor muscles to support the growing belly and prepare for labor. Avoid Certain Exercises: Avoid exercises that involve lying flat on the back after the first trimester, deep squats, and heavy lifting.\n\nFlexibility and Balance:\n\nPrenatal Yoga: Engage in prenatal yoga, which includes modified poses that accommodate the changing body and focus on relaxation and breathing techniques. Stretching: Perform gentle stretching exercises to maintain flexibility. Avoid overstretching and positions that strain the joints.\n\nHigh-Intensity Interval Training (HIIT):\n\nModify Intensity and Movements: Reduce the intensity and avoid high-impact movements. Focus on low-impact variations and shorter intervals. Monitor Heart Rate: Keep heart rate within a safe range as advised by a healthcare provider.\n\nBest Recommendation:\n\nEngage in low-impact aerobic activities such as walking and swimming. Incorporate modified strength training focusing on lighter weights and more repetitions. Practice prenatal yoga for flexibility, relaxation, and stress management. Avoid high-risk activities and exercises that involve lying flat on the back after the first trimester.";
        break;
      default:
        botResponse = "I'm sorry, I didn't understand that. Can you please ask a different question?";
    }

    setMessages((prevMessages) => [
      ...prevMessages,
      { type: 'bot', text: botResponse },
    ]);

    setLoading(false); // Hide loading indicator
  };

  const renderItem = ({ item }) => (
    <View style={[styles.messageContainer, item.type === 'user' ? styles.userMessage : styles.botMessage]}>
      <Text style={styles.messageText}>{item.text}</Text>
    </View>
  );

  const scrollToBottom = () => {
    if (flatListRef.current) {
      flatListRef.current.scrollToEnd({ animated: true });
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('Exercise')}>
          <Ionicons name="arrow-back" size={24} color="#007BFF" />
        </TouchableOpacity>
        <Text style={styles.headerText}>Ask Coach</Text>
      </View>
      <View style={styles.options}>
        {/* The buttons are removed as manual input is required */}
      </View>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={(item, index) => index.toString()}
        style={styles.messageList}
        onContentSizeChange={scrollToBottom} // Automatically scroll to bottom when content size changes (new message)
      />
      {loading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#007BFF" />
        </View>
      )}
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask me..."
        />
        <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
          <Ionicons name="send-sharp" size={25} color="white" />
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    justifyContent: 'space-between',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#ccc',
  },
  backButton: {
    paddingVertical: 10,
    paddingHorizontal: 10,
  },
  headerText: {
    fontSize: 24,
    fontWeight: 'bold',
    textAlign: 'center',
    flex: 1,
  },
  optionButton: {
    width: 150,
    height: 50,
    backgroundColor: '#F1F0F0',
    paddingVertical: 10,
    paddingHorizontal: 15,
    borderRadius: 10,
    marginVertical: 5,
    marginHorizontal: 5,
    alignItems: 'center',
    borderColor: '#4285F4',
    borderWidth: 1,
  },
  optionText: {
    fontSize: 14,
  },
  messageList: {
    flex: 1,
  },
  messageContainer: {
    padding: 10,
    borderRadius: 5,
    marginVertical: 5,
    maxWidth: '80%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#DCF8C6',
  },
  botMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#F1F0F0',
  },
  messageText: {
    fontSize: 16,
  },
  inputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    borderTopWidth: 1,
    borderColor: '#ccc',
    padding: 10,
  },
  input: {
    flex: 1,
    height: 40,
    borderColor: '#ccc',
    borderWidth: 1,
    borderRadius: 5,
    paddingHorizontal: 10,
    marginRight: 10,
  },
  sendButton: {
    backgroundColor: '#007BFF',
    paddingVertical: 10,
    paddingHorizontal: 20,
    borderRadius: 5,
  },
  sendButtonText: {
    color: '#fff',
    fontSize: 16,
  },
  options: {
    flexDirection: 'column',
    justifyContent: 'space-around',
    marginVertical: 10,
  },
  loadingContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginVertical: 10,
  },
});

export default Chatbot;




/////////////////////////Code For API End Points /////////////////////////////
// import React, { useState, useRef, useEffect } from 'react';
// import { View, Text, TextInput, TouchableOpacity, StyleSheet, FlatList, ActivityIndicator } from 'react-native';
// import Ionicons from '@expo/vector-icons/Ionicons';
// import { useNavigation } from '@react-navigation/native';

// const Chatbot = () => {
//   const [messages, setMessages] = useState([]);
//   const [input, setInput] = useState('');
//   const [loading, setLoading] = useState(false);
//   const flatListRef = useRef(null);
//   const navigation = useNavigation(); // Hook to get navigation instance

//   useEffect(() => {
//     console.log("Updated Messages:", messages); // Log messages whenever they update
//   }, [messages]);

//   const handleOptionSelect = async (option) => {
//     await fetchBotResponse(option);
//     setInput('');
//     scrollToBottom(); // Scroll to the bottom when a new message is added
//   };

//   const handleSend = async () => {
//     if (input.trim()) {
//       const userMessage = input;
//       setInput('');
//       await fetchBotResponse(userMessage);
//       scrollToBottom(); // Scroll to the bottom when a new message is added
//     }
//   };

//   const fetchBotResponse = async (prompt) => {
//     setMessages((prevMessages) => [...prevMessages, { type: 'user', text: prompt }]);
//     setLoading(true); // Show loading indicator

//     try {
//       const response = await fetch('https://formfit-49a7aabb3be8.herokuapp.com/api/predict', {
//         method: 'POST',
//         headers: {
//           'Content-Type': 'application/json',
//         },
//         body: JSON.stringify({ prompt }),
//       });

//       if (!response.ok) {
//         throw new Error('Network response was not ok');
//       }

//       const data = await response.json();
//       console.log("The Bot data is :", data);

//       setMessages((prevMessages) => [
//         ...prevMessages,
//         { type: 'bot', text: data.bot }, // Correct property accessed here
//       ]);
//     } catch (error) {
//       console.error('Error:', error);
//       setMessages((prevMessages) => [
//         ...prevMessages,
//         { type: 'bot', text: 'Error fetching response. Please try again later.' },
//       ]);
//     } finally {
//       setLoading(false); // Hide loading indicator
//     }
//   };

//   const renderItem = ({ item }) => (
//     <View style={[styles.messageContainer, item.type === 'user' ? styles.userMessage : styles.botMessage]}>
//       <Text style={styles.messageText}>{item.text}</Text>
//     </View>
//   );

//   const scrollToBottom = () => {
//     if (flatListRef.current) {
//       flatListRef.current.scrollToEnd({ animated: true });
//     }
//   };

//   return (
//     <View style={styles.container}>
//       <View style={styles.header}>
//         <TouchableOpacity style={styles.backButton} onPress={() => navigation.navigate('Exercise')}>
//           <Ionicons name="arrow-back" size={24} color="#007BFF" />
//         </TouchableOpacity>
//         <Text style={styles.headerText}>Ask Coach</Text>
//       </View>
//       <View style={styles.options}>
//         <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Upper body')}>
//           <Text style={styles.optionText}>Upper body</Text>
//         </TouchableOpacity>
//         <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Core')}>
//           <Text style={styles.optionText}>Core</Text>
//         </TouchableOpacity>
//         <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Lower body')}>
//           <Text style={styles.optionText}>Lower body</Text>
//         </TouchableOpacity>
//       </View>
//       <FlatList
//         ref={flatListRef}
//         data={messages}
//         renderItem={renderItem}
//         keyExtractor={(item, index) => index.toString()}
//         style={styles.messageList}
//         onContentSizeChange={scrollToBottom} // Automatically scroll to bottom when content size changes (new message)
//       />
//       {loading && (
//         <View style={styles.loadingContainer}>
//           <ActivityIndicator size="large" color="#007BFF" />
//         </View>
//       )}
//       <View style={styles.inputContainer}>
//         <TextInput
//           style={styles.input}
//           value={input}
//           onChangeText={setInput}
//           placeholder="Ask me..."
//         />
//         <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
//           <Ionicons name="send-sharp" size={25} color="white" />
//         </TouchableOpacity>
//       </View>
//     </View>
//   );
// };

// const styles = StyleSheet.create({
//   container: {
//     flex: 1,
//     backgroundColor: '#fff',
//     justifyContent: 'space-between',
//   },
//   header: {
//     flexDirection: 'row',
//     alignItems: 'center',
//     padding: 10,
//     borderBottomWidth: 1,
//     borderBottomColor: '#ccc',
//   },
//   backButton: {
//     paddingVertical: 10,
//     paddingHorizontal: 10,
//   },
//   headerText: {
//     fontSize: 24,
//     fontWeight: 'bold',
//     textAlign: 'center',
//     flex: 1,
//   },
//   optionButton: {
//     width: 150,
//     height: 50,
//     backgroundColor: '#F1F0F0',
//     paddingVertical: 10,
//     paddingHorizontal: 15,
//     borderRadius: 10,
//     marginVertical: 5,
//     marginHorizontal: 5,
//     alignItems: 'center',
//     borderColor: '#4285F4',
//     borderWidth: 1,
//   },
//   optionText: {
//     fontSize: 14,
//   },
//   messageList: {
//     flex: 1,
//   },
//   messageContainer: {
//     padding: 10,
//     borderRadius: 5,
//     marginVertical: 5,
//     maxWidth: '80%',
//   },
//   userMessage: {
//     alignSelf: 'flex-end',
//     backgroundColor: '#DCF8C6',
//   },
//   botMessage: {
//     alignSelf: 'flex-start',
//     backgroundColor: '#F1F0F0',
//   },
//   messageText: {
//     fontSize: 16,
//   },
//   inputContainer: {
//     flexDirection: 'row',
//     alignItems: 'center',
//     borderTopWidth: 1,
//     borderColor: '#ccc',
//     padding: 10,
//   },
//   input: {
//     flex: 1,
//     height: 40,
//     borderColor: '#ccc',
//     borderWidth: 1,
//     borderRadius: 5,
//     paddingHorizontal: 10,
//     marginRight: 10,
//   },
//   sendButton: {
//     backgroundColor: '#007BFF',
//     paddingVertical: 10,
//     paddingHorizontal: 20,
//     borderRadius: 5,
//   },
//   sendButtonText: {
//     color: '#fff',
//     fontSize: 16,
//   },
//   options: {
//     flexDirection: 'column',
//     justifyContent: 'space-around',
//     marginVertical: 10,
//   },
//   loadingContainer: {
//     flexDirection: 'row',
//     justifyContent: 'center',
//     marginVertical: 10,
//   },
// });

// export default Chatbot;

