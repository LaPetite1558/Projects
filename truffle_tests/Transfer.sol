pragma solidity ^0.4.22;

contract Transfer {
	
	mapping (address => uint256) public balances;
		
	function deposit() public payable {
		balances[msg.sender] += msg.value;
	}
	
	function withdraw(uint value) public {
		require(balances[msg.sender] > value);
		balances[msg.sender] -= value;
		msg.sender.transfer(value);
	}
	
	function transfer(uint value, address to) public {
		require(balances[msg.sender] > value);
		balances[msg.sender] -= value;
		to.transfer(value);
	}
}