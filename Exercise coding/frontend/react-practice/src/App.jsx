import './App.css'
import PersonalInfo from './components/PersonalInfo';
import UserCard from './src/components/UserCard';

function App() {
  return (
    <div className="App">
      <PersonalInfo />
      <UserCard name="John Doe" email="john@example.com" role="Developer" />
      <UserCard name="Jane Smith" email="jane@example.com" role="Designer" />
    </div>
  )
}

export default App