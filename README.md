# Intelligent Inquiry Solution

Developing an FAQs Chatbot for Efficient Management of University Admission Inquiries at Dayalbagh Educational Institute.

## Objectives

- Save effort and time for both the admission and registration staff and students who wish to enroll.
- Provide detailed information about colleges and majors.
- Easy access to information.

## Scope

- People who wish to enroll at Dayalbagh Educational Institute
- Admission and registration staff.

## Abstract

This project develops a specialized chatbot for admissions at D.E.I. College, leveraging AI to streamline processes, enhance accessibility, and set a precedent for transformative technology in education.

## Introduction

In the age of rapid technological advancements, chatbots, powered by artificial intelligence, are reshaping interactions between institutions and stakeholders. This project focuses on developing a specialized chatbot for admission processes at Dayalbagh Educational Institute (D.E.I.) College. By leveraging AI, the chatbot aims to streamline admissions, providing instant access to essential information for new applicants. This initiative highlights the transformative potential of AI-driven technologies in education and beyond.

## Need of Solution

The surge in admission inquiries overwhelms offices, necessitating an FAQs Chatbot solution to efficiently manage queries. This addresses several needs: handling high volumes of inquiries, optimizing resource allocation, providing instant responses, ensuring privacy, and enhancing the overall admission experience.

## User Groups

In developing an FAQs Chatbot for addressing admission inquiries at Dayalbagh Educational Institute (DEI), it is imperative to identify the various user groups that will interact with or benefit from the system.

1. Prospective Students and Parents
2. Admission Office Staff
3. Faculty and Academic Advisors
4. Administrative Staff
5. Alumni and Current Students

These user groups rely on the FAQs Chatbot to streamline admission inquiries, support prospective students, and uphold admission service standards at DEI, leveraging its capabilities to efficiently address inquiries and enhance the overall experience.

## Attributes

Accurate Information Retrieval
Prompt Response Time
Comprehensive Coverage
Memory Storage
User-Friendly Interface
Cost-Effectiveness
Availability
Automated Inquiry Handling
Integration with Existing Systems
These attributes guide the development process to ensure the Chatbot meets critical user needs, maintains trust, and aligns with admission service standards at DEI.

## System Requirements



### Functional Requirements

1. Clear information about Admission policy
2. Detailed information about university colleges
3. Detailed information about colleges’ programs and majors
4. Clarification of permitted secondary school branches and minimum CPAs
5. Duration of study and parallel study policy for each major
6. Graduation plans and placement tests information
7. First installment costs and credit hour prices for each major

### Non-Functional Requirements

1. Handling multiple user inputs without long wait times
2. Delayed responses to simulate human interaction
3. Appropriate and tuned data set for accurate responses
4. Data training focused on Admission and Registration deanship content
5. Prevention of abusive language
6. Ability to extend the project to include all colleges


## Models and interfaces

1. Natural Language Processing (NLP) Model:
   - Selection of NLP framework (e.g., Haystack)
   - Use of pre-trained language models (e.g., Lamini)
   - Fine-tuning approach for model training
   - Handling multilingual support, initially focusing on English and Hindi

2. User Interfaces:
   - Website Integration

3. Voice Interface (Optional):
   - Potential consideration for future phases to enhance accessibility

4. Data Integration:
   - Integration with university databases for accurate information retrieval

5. User Authentication (Optional):
   - Potential implementation for personalized responses and secure data handling

6. Multilingual Support:
   - Designed to support multiple languages for broader user accessibility


## System Demonstration

### 1. Admin Panel

In this section, I provide a detailed account of the system demonstration conducted for the interface designed to manage colleges and majors within the Deanship of Admission and Registration at D.E.I. University. The demonstration encompassed key aspects, including implementation details, challenges encountered during implementation, snapshots of the user interface, and a comprehensive description of the user interface.

### 2. Implementation Details

The demonstration commenced with a comprehensive overview of the interface's implementation details, as follows:

#### 2.1 Access and Authentication

- The demonstration illustrated the secure login process for authorized users.
- The importance of authentication in ensuring secure access was emphasized.

#### 2.2 Major Management

- Adding new majors within a college was demonstrated, highlighting major-specific details.
- The process of amending major information was shown, ensuring immediate updates.
- Deleting majors was presented, with confirmation mechanisms to prevent inadvertent data loss.

### 3. Implementation Issues

The demonstration acknowledged and discussed certain implementation challenges that were encountered during the development of the interface. These challenges were effectively addressed and resolved, underscoring the significance of ongoing maintenance and updates.

### 4. Snapshots and UI Description

- **Login**: This page is the first page to be shown when accessing the site.

   ![WhatsApp Image 2024-03-22 at 21 40 14_418130a0](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/f3417ef1-9003-453b-a4c1-1b1c110952c6)

- **Dashboard**: This page will be the first thing that appears to the user when he logs into the site correctly, whether he is an administrator or a viewer.

  
  ![WhatsApp Image 2024-03-22 at 21 43 04_a723cb13](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/60059a72-8967-4c01-b5ba-3b107be63dc4)

  ![WhatsApp Image 2024-03-22 at 21 43 17_11cf19e2](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/4e180218-8b91-440e-ba3e-4c29c8f1ea00)

### Chatbot Implementation Details

1. **Streamlit UI**: User interface developed using Streamlit for intuitive interaction with the chatbot.

2. **Python**: Primary programming language handling data processing, interaction with Streamlit, and integration with Haystack.

3. **Haystack**: Open-source framework simplifying search and question-answering applications, utilized for information retrieval and processing.

4. **Lamini**: Component within Haystack framework responsible for natural language understanding and information retrieval.

5. **QnA FAQ Excel Dataset**: Custom dataset containing FAQs and admission details, serving as the chatbot's knowledge base for accurate responses.

This comprehensive technology stack ensures effective understanding of user queries and provision of accurate responses, reflecting the commitment to a sophisticated chatbot tailored to D.E.I. College's admission process.
![WhatsApp Image 2024-03-22 at 21 27 43_8ed0a714](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/645b6d3c-f9b2-41a0-b62d-36fbed55d317)

### Challenges Encountered

During each stage of development, I encountered numerous challenges, some of which took days to solve. The most significant issues I faced were:

 **Language Support and Translation Accuracy**:

Supporting multiple languages, including Hindi, presented a challenge. Ensuring accurate translation and understanding of language nuances was vital for providing meaningful responses.

**Data Collection and Quality**:

Gathering a comprehensive dataset that included accurate and culturally relevant information in Hindi was time-consuming. Maintaining data quality and consistency was crucial for the chatbot's performance.

**Scalability**:

As the user base grew, the chatbot had to handle increased demand. Ensuring the system could scale effectively to accommodate a growing number of users was a technical challenge.

**Data Privacy and Security**:

Handling sensitive admission-related information required strong security measures to protect user data and comply with relevant privacy regulations.

**Knowledge Base Maintenance**:

As the admission process evolved and information changed, maintaining and updating the chatbot's knowledge base was an ongoing challenge.

### Streamlit UI Description

- **User-Friendly Interface**: Designed with a visually appealing layout for easy interaction.
- **Chat Box**: Core interface element allowing users to type or speak questions.
- **Multilingual Support**: Seamless support for English and Hindi.
- **Search Functionality**: Included for quick access to specific information.
- **Responsive Design**: Ensures effective functionality across devices.
- **Help and Documentation**: Dedicated section provides guidance on chatbot usage.
- **Privacy and Security Information**: Accessible details reassure users about data handling.

### Important Information

- When training the bot, it is important to ensure that the structure in the domain file is in the right format or it will fail.
- Training time is machine dependent.
- It's okay to delete old models, but it's not recommended.

### Output Screenshot
![WhatsApp Image 2024-03-23 at 08 20 30_016a6cf7](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/2a876a6d-a3da-48e1-a111-6c78e406aa06)
![WhatsApp Image 2024-03-23 at 08 20 39_1926cdbb](https://github.com/SuratPyari18/Intelligent-Inquiry-Solution/assets/164517462/b188ea7b-d3e5-41da-bcc9-856571df16ad)

## Conclusion

In the future development and enhancement of the D.E.I. College Admission Chatbot, I anticipate encountering various challenges and making interesting decisions as I continue to refine and optimize this valuable tool. These challenges will be opportunities for growth, and my decisions will shape the chatbot's performance. As I move forward, I can outline some of the key aspects that will be pivotal in the chatbot's evolution:

### Challenges:

- **Multilingual Adaptation**: Addressing the nuances and intricacies of multiple languages, particularly Hindi, will be a challenge. Future efforts will focus on enhancing language support and ensuring precise translations.
- **User Feedback Integration**: Harnessing user feedback will be a cornerstone of improvement. Future initiatives will center on developing robust mechanisms for collecting, analyzing, and utilizing feedback to enhance the chatbot's performance.
- **Scalability**: Anticipating the growth in user demand, future considerations will include further scalability to accommodate increasing user loads without compromising response times and quality.

### Interesting Decisions:

- **Continued Model Enhancement**: As technology evolves, I will continue to explore new models and techniques to ensure that my chatbot remains at the forefront of natural language processing capabilities.
- **Enhanced Multimodal Support**: Embracing and implementing advanced techniques for handling multimodal inputs, such as text, images, and voice, will be an intriguing decision to enrich user interactions.
- **Adaptive Contextual Learning**: We will delve into developing adaptive learning capabilities, enabling the chatbot to evolve based on user interactions and context.

## Limitations:
 The chatbot will only answer questions included in the dataset, operating on the principle of vector search to retrieve and respond to inputs. While this approach conserves memory and functions without internet connectivity, it is constrained by the dataset's content and may not address queries outside its scope.
