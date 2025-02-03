import random
from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict
import heapq
from enum import Enum
import time
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich import box
from rich.panel import Panel
from rich.layout import Layout
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

class CallPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class CallType(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    GENERAL = "general"

@dataclass
class Call:
    id: int
    arrival_time: float
    priority: CallPriority
    call_type: CallType
    duration: float
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    assigned_agent: Optional[str] = None
    waiting_time: Optional[float] = None

@dataclass
class Agent:
    id: str
    skills: List[CallType]
    efficiency_factor: float
    current_call: Optional[Call] = None
    total_calls_handled: int = 0
    total_handle_time: float = 0
    call_history: List[Call] = None

    def __post_init__(self):
        self.call_history = []

class CallCenterSimulation:
    def __init__(
        self,
        num_agents: int,
        simulation_duration: float,
        mean_call_arrival_rate: float,
        mean_call_duration: float,
        real_time_factor: float = 0.1  # Controls animation speed
    ):
        self.console = Console()
        self.current_time = 0.0
        self.simulation_duration = simulation_duration
        self.mean_call_arrival_rate = mean_call_arrival_rate
        self.mean_call_duration = mean_call_duration
        self.real_time_factor = real_time_factor
        
        # Initialize agents
        self.agents = []
        for i in range(num_agents):
            skills = list(CallType)
            efficiency = random.uniform(0.8, 1.2)
            self.agents.append(Agent(f"Agent_{i}", skills, efficiency))
        
        self.call_queue = deque()
        self.events = []
        self.call_counter = 0
        
        # Enhanced statistics
        self.completed_calls = []
        self.abandoned_calls = 0
        self.total_wait_time = 0.0
        self.hourly_stats = []
        self.queue_length_history = []
        self.agent_utilization = {agent.id: [] for agent in self.agents}
        
        self._schedule_next_arrival()
    def _schedule_next_arrival(self):
        """Schedules the next call arrival based on an exponential distribution."""
        interarrival_time = random.expovariate(self.mean_call_arrival_rate)
        next_arrival = self.current_time + interarrival_time

        if next_arrival < self.simulation_duration:
            call = self._generate_call(next_arrival)
            heapq.heappush(self.events, (next_arrival, "arrival", call))


    def log_call(self, call: Call):
        with open("call_log.txt", "a") as f:
            f.write(f"Call {call.id} - {call.priority.name} - {call.call_type.name} - Duration: {call.duration:.2f} minutes - Waiting Time: {call.waiting_time:.2f} minutes\n")

    def check_queue_alert(self):
        if len(self.call_queue) > 5:
            self.console.print("[bold red]Alert: Queue length exceeded 5 calls![/]")
    
    def run(self):
        self.console.print("[bold green]Starting Call Center Simulation...[/]")
        
        with Live(self._create_live_display(), refresh_per_second=4) as live:
            while self.events:
                current_event_time, event_type, call = heapq.heappop(self.events)
                self.current_time = current_event_time
                
                if event_type == "arrival":
                    call.waiting_time = self.current_time - call.arrival_time
                    self._handle_call_arrival(call)
                    self._schedule_next_arrival()
                    self.log_call(call)
                elif event_type == "completion":
                    self._handle_call_completion(call)
                
                self._update_hourly_stats()
                self.queue_length_history.append((self.current_time, len(self.call_queue)))
                self.check_queue_alert()
                
                for agent in self.agents:
                    utilization = 1 if agent.current_call else 0
                    self.agent_utilization[agent.id].append((self.current_time, utilization))
                
                live.update(self._create_live_display())
                time.sleep(self.real_time_factor)
    
    def export_calls_to_csv(self, filename="call_data.csv"):
        if not self.completed_calls:
            self.console.print("[bold red]No completed calls to export![/]")
            return
        df = pd.DataFrame([call.__dict__ for call in self.completed_calls])
        df.to_csv(filename, index=False)
        self.console.print(f"[bold blue]Call data exported to {filename}[/]")


    def generate_agent_utilization_report(self, filename="agent_utilization.png"):
        if not any(self.agent_utilization.values()):
            self.console.print("[bold red]No agent utilization data to plot![/]")
            return
        plt.figure(figsize=(10, 5))
        for agent_id, data in self.agent_utilization.items():
            if data:
                times, utilizations = zip(*data)
                plt.plot(times, utilizations, label=agent_id)
        plt.xlabel("Time (minutes)")
        plt.ylabel("Utilization (1 = Busy, 0 = Available)")
        plt.title("Agent Utilization Over Time")
        plt.legend()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.console.print(f"[bold blue]Agent utilization graph saved as {filename}[/]")


    def _generate_call(self, arrival_time: float) -> Call:
        self.call_counter += 1
        priority = random.choices(
            list(CallPriority),
            weights=[0.7, 0.2, 0.1]
        )[0]
        call_type = random.choice(list(CallType))
        duration = random.expovariate(1.0 / self.mean_call_duration)
        
        return Call(
            id=self.call_counter,
            arrival_time=arrival_time,
            priority=priority,
            call_type=call_type,
            duration=duration
        )

    def _find_available_agent(self, call: Call) -> Optional[Agent]:
        available_agents = [
            agent for agent in self.agents
            if not agent.current_call and call.call_type in agent.skills
        ]
        
        if available_agents:
            return min(available_agents, key=lambda a: a.total_calls_handled)
        return None

    def _handle_call_arrival(self, call: Call):
        agent = self._find_available_agent(call)
        
        if agent:
            agent.current_call = call
            call.start_time = self.current_time
            call.assigned_agent = agent.id
            
            adjusted_duration = call.duration * agent.efficiency_factor
            completion_time = self.current_time + adjusted_duration
            heapq.heappush(self.events, (completion_time, "completion", call))
        else:
            self.call_queue.append(call)

    def _handle_call_completion(self, call: Call):
        agent = next(a for a in self.agents if a.id == call.assigned_agent)
        
        call.end_time = self.current_time
        agent.total_calls_handled += 1
        agent.total_handle_time += (call.end_time - call.start_time)
        agent.call_history.append(call)
        agent.current_call = None
        self.completed_calls.append(call)
        
        if self.call_queue:
            next_call = self.call_queue.popleft()
            self._handle_call_arrival(next_call)

    def _update_hourly_stats(self):
        current_hour = int(self.current_time / 60)
        while len(self.hourly_stats) <= current_hour:
            self.hourly_stats.append({
                'hour': len(self.hourly_stats),
                'calls_handled': 0,
                'avg_wait_time': 0,
                'avg_handle_time': 0,
                'queue_length': len(self.call_queue)
            })
        
        current_stats = self.hourly_stats[current_hour]
        hour_calls = [
            call for call in self.completed_calls
            if int(call.end_time / 60) == current_hour
        ]
        
        current_stats['calls_handled'] = len(hour_calls)
        
        if hour_calls:
            current_stats['avg_wait_time'] = np.mean([
                call.start_time - call.arrival_time for call in hour_calls
            ])
            current_stats['avg_handle_time'] = np.mean([
                call.end_time - call.start_time for call in hour_calls
            ])

    def _create_live_display(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer")
        )
        
        # Header
        header = Table.grid()
        header.add_column(style="bold magenta", justify="center")
        header.add_row("🎧 Call Center Simulation Live Monitor 📞")
        header.add_row(f"Time: {self.current_time:.1f} minutes")
        
        # Main content
        main_table = Table(
            show_header=True,
            header_style="bold blue",
            box=box.ROUNDED
        )
        main_table.add_column("Metric")
        main_table.add_column("Value")
        
        active_calls = sum(1 for agent in self.agents if agent.current_call)
        queue_length = len(self.call_queue)
        
        
        main_table.add_row(
            "Active Calls",
            f"[green]{active_calls}[/]"
        )
        main_table.add_row(
            "Queue Length",
            f"[{'red' if queue_length > 5 else 'green'}]{queue_length}[/]"
        )
        main_table.add_row(
            "Completed Calls",
            f"[blue]{len(self.completed_calls)}[/]"
        )
        
        
        footer = Table(
            show_header=True,
            header_style="bold green",
            box=box.ROUNDED
        )
        footer.add_column("Agent")
        footer.add_column("Status")
        footer.add_column("Calls Handled")
        footer.add_column("Efficiency")
        
        for agent in self.agents:
            status_emoji = "🟢" if not agent.current_call else "🔴"
            status_text = "Available" if not agent.current_call else f"Call #{agent.current_call.id}"
            footer.add_row(
                f"[cyan]{agent.id}[/]",
                f"{status_emoji} {status_text}",
                str(agent.total_calls_handled),
                f"{agent.efficiency_factor:.2f}"
            )
        
        # Update layout
        layout["header"].update(Panel(header))
        layout["main"].update(Panel(main_table))
        layout["footer"].update(Panel(footer))
        
        return layout

    def run(self):
        self.console.print("[bold green]Starting Call Center Simulation...[/]")
        
        with Live(self._create_live_display(), refresh_per_second=4) as live:
            while self.events:
                current_event_time, event_type, call = heapq.heappop(self.events)  \
                self.current_time = current_event_time
                
                if event_type == "arrival":
                    self._handle_call_arrival(call)
                    self._schedule_next_arrival()
                elif event_type == "completion":
                    self._handle_call_completion(call)
                
                self._update_hourly_stats()
                self.queue_length_history.append((self.current_time, len(self.call_queue)))
                
                live.update(self._create_live_display())
                time.sleep(self.real_time_factor)

    def generate_reports(self):
        self.console.print("\n[bold magenta]Generating Analysis Reports...[/]")
        
        
        calls_df = pd.DataFrame([
            {
                'id': call.id,
                'arrival_time': call.arrival_time,
                'wait_time': call.start_time - call.arrival_time,
                'handle_time': call.end_time - call.start_time,
                'priority': call.priority.name,
                'type': call.call_type.name,
                'agent': call.assigned_agent
            }
            for call in self.completed_calls
        ])
        
        
        plt.style.use('ggplot')  
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        
        calls_df['hour'] = calls_df['arrival_time'].apply(lambda x: int(x/60))
        hourly_calls = calls_df.groupby('hour').size()
        axes[0, 0].bar(hourly_calls.index, hourly_calls.values, color='skyblue')
        axes[0, 0].set_title('Call Volume by Hour')
        axes[0, 0].set_xlabel('Hour')
        axes[0, 0].set_ylabel('Number of Calls')
        
        
        priority_wait_times = calls_df.groupby('priority')['wait_time'].mean()
        axes[0, 1].bar(priority_wait_times.index, priority_wait_times.values, color='lightgreen')
        axes[0, 1].set_title('Average Wait Times by Priority')
        axes[0, 1].set_ylabel('Wait Time (minutes)')
        
       
        queue_times, queue_lengths = zip(*self.queue_length_history)
        axes[1, 0].plot(queue_times, queue_lengths, color='orange', linewidth=2)
        axes[1, 0].fill_between(queue_times, queue_lengths, alpha=0.3, color='orange')
        axes[1, 0].set_title('Queue Length Over Time')
        axes[1, 0].set_xlabel('Time (minutes)')
        axes[1, 0].set_ylabel('Queue Length')
        
        
        agent_stats = pd.DataFrame([
            {
                'agent': agent.id,
                'calls_handled': agent.total_calls_handled,
                'avg_handle_time': agent.total_handle_time / agent.total_calls_handled
                if agent.total_calls_handled > 0 else 0
            }
            for agent in self.agents
        ])
        
        axes[1, 1].scatter(
            agent_stats['calls_handled'],
            agent_stats['avg_handle_time'],
            color='purple',
            alpha=0.6,
            s=100
        )
        axes[1, 1].set_title('Agent Performance')
        axes[1, 1].set_xlabel('Calls Handled')
        axes[1, 1].set_ylabel('Avg Handle Time (min)')
        
        plt.tight_layout()
        plt.savefig('call_center_analysis.png', dpi=300, bbox_inches='tight')
        
        
        self.console.print("\n[bold green]Summary Statistics:[/]")
        
        summary_table = Table(show_header=True, header_style="bold blue", box=box.ROUNDED)
        summary_table.add_column("Metric")
        summary_table.add_column("Value")
        
        total_wait_time = sum(call.start_time - call.arrival_time for call in self.completed_calls)
        total_handle_time = sum(call.end_time - call.start_time for call in self.completed_calls)
        
        if self.completed_calls:
            avg_wait_time = total_wait_time / len(self.completed_calls)
            avg_handle_time = total_handle_time / len(self.completed_calls)
        else:
            avg_wait_time = 0
            avg_handle_time = 0
        
        summary_stats = {
            "Total Calls Completed": len(self.completed_calls),
            "Average Wait Time": f"{avg_wait_time:.2f} minutes",
            "Average Handle Time": f"{avg_handle_time:.2f} minutes",
            "Maximum Queue Length": max((len_ for _, len_ in self.queue_length_history), default=0),
            "Abandoned Calls": self.abandoned_calls
        }
        
        for metric, value in summary_stats.items():
            summary_table.add_row(metric, str(value))
        
        self.console.print(summary_table)

def main():
    
    params = {
        "num_agents": 5,
        "simulation_duration": 480, 
        "mean_call_arrival_rate": 0.2, 
        "mean_call_duration": 10, 
        "real_time_factor": 0.1  
    }
    
    
    console = Console()
    
    with console.status("[bold green]Initializing simulation...") as status:
        sim = CallCenterSimulation(**params)
        
        console.print("[bold blue]Simulation parameters:[/]")
        for param, value in params.items():
            console.print(f"  {param}: {value}")
        
        time.sleep(1)
        console.print("\n[bold green]Starting simulation...[/]")
        
    
    sim.run()
    
    
    sim.generate_reports()
    
    console.print("\n[bold green]Simulation completed! Reports have been generated.[/]")
    console.print("[bold blue]Check 'call_center_analysis.png' for visualizations.[/]")

if __name__ == "__main__":
    main()
