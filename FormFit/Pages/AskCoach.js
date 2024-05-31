import React, { useState, useRef } from 'react';
import { View, Text, TextInput, TouchableOpacity, StyleSheet, FlatList } from 'react-native';
import Ionicons from '@expo/vector-icons/Ionicons';
import { useNavigation } from '@react-navigation/native';

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const flatListRef = useRef(null);
  const navigation = useNavigation(); // Hook to get navigation instance

  const handleOptionSelect = (option) => {
    const response = getBotResponse(option);
    setMessages([...messages, { type: 'user', text: option }, { type: 'bot', text: response }]);
    setInput('');
    scrollToBottom(); // Scroll to the bottom when a new message is added
  };

  const handleSend = () => {
    if (input.trim()) {
      const response = getBotResponse(input);
      setMessages([...messages, { type: 'user', text: input }, { type: 'bot', text: response }]);
      setInput('');
      scrollToBottom(); // Scroll to the bottom when a new message is added
    }
  };

  const getBotResponse = (prompt) => {
    switch (prompt.toLowerCase()) {
      case 'upper body':
        return 'Upper body exercises include push-ups, pull-ups, and shoulder presses.';
      case 'core':
        return 'Core exercises include planks, sit-ups, and Russian twists.';
      case 'lower body':
        return 'Lower body exercises include squats, lunges, and deadlifts.';
      default:
        return 'This is a static response to your prompt: ' + prompt;
    }
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
        <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Upper body')}>
          <Text style={styles.optionText}>Upper body</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Core')}>
          <Text style={styles.optionText}>Core</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.optionButton} onPress={() => handleOptionSelect('Lower body')}>
          <Text style={styles.optionText}>Lower body</Text>
        </TouchableOpacity>
      </View>
      <FlatList
        ref={flatListRef}
        data={messages}
        renderItem={renderItem}
        keyExtractor={(item, index) => index.toString()}
        style={styles.messageList}
        onContentSizeChange={scrollToBottom} // Automatically scroll to bottom when content size changes (new message)
      />
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder="Ask me..."
        />
        <TouchableOpacity style={styles.sendButton} onPress={handleSend}>
          <Text style={styles.sendButtonText}><Ionicons name="send-sharp" size={25} color="white" /></Text>
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
});

export default Chatbot;
